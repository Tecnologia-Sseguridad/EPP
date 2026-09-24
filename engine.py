"""Local facial engine. Workers exchange only latest immutable frame/results."""
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import queue
import sqlite3
import threading
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_ID = "sface-2021dec-opencv-128-v1"


def classify_trial(expected, accepted, observations):
    if observations == 0:
        return "sin_muestra_valida"
    if expected == "(Desconocido)":
        return "falsa_aceptacion" if accepted else "rechazo_correcto"
    if accepted - {expected}:
        return "identidad_incorrecta"
    return "identificacion_correcta" if expected in accepted else "falso_rechazo"


class Latest:
    def __init__(self):
        self.lock = threading.Lock()
        self.value = None

    def put(self, value):
        with self.lock:
            self.value = value

    def get(self):
        with self.lock:
            return self.value


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float32).flatten()
    norm = np.linalg.norm(vector)
    if not np.isfinite(vector).all() or norm < 1e-8:
        raise ValueError("Vector facial inválido")
    return vector / norm


def match(vector, gallery, threshold=0.45, margin=0.08):
    """Compare best template per identity; runner-up must be another identity."""
    if not gallery:
        return None, 0.0, 0.0
    scores = sorted(((name, float(np.max(templates @ vector))) for name, templates in gallery.items()), key=lambda row: row[1], reverse=True)
    name, score = scores[0]
    gap = score - scores[1][1] if len(scores) > 1 else 2.0
    return (name if score >= threshold and gap >= margin else None), score, gap


class Store:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS templates (name TEXT NOT NULL, model TEXT NOT NULL, vector BLOB NOT NULL)")
        self.db.commit()

    def save(self, name, vectors):
        # Replace all samples atomically only after a complete enrolment.
        with self.db:
            self.db.execute("DELETE FROM templates WHERE name=? AND model=?", (name, MODEL_ID))
            self.db.executemany("INSERT INTO templates VALUES (?, ?, ?)", [(name, MODEL_ID, normalize(v).tobytes()) for v in vectors])

    def gallery(self):
        result = {}
        for name, blob in self.db.execute("SELECT name, vector FROM templates WHERE model=?", (MODEL_ID,)):
            vector = np.frombuffer(blob, dtype=np.float32)
            if vector.size != 128:
                raise ValueError("Plantilla incompatible con SFace")
            result.setdefault(name, []).append(normalize(vector))
        return {name: np.stack(vectors) for name, vectors in result.items()}

    def close(self):
        self.db.close()


class Vision:
    def __init__(self):
        for name in ("yunet.onnx", "sface.onnx"):
            if not (ROOT / "models" / name).exists():
                raise FileNotFoundError("Faltan modelos. Ejecuta: .venv\\Scripts\\python.exe setup_models.py")
        cv2.setNumThreads(2)
        self.detector = cv2.FaceDetectorYN.create(str(ROOT / "models/yunet.onnx"), "", (640, 360), 0.85, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(ROOT / "models/sface.onnx"), "")
        self.detector.detect(np.zeros((360, 640, 3), np.uint8))
        self.recognizer.feature(np.zeros((112, 112, 3), np.uint8))

    def analyze(self, frame):
        height, width = frame.shape[:2]
        scale = min(640 / width, 1.0)
        small = cv2.resize(frame, (round(width * scale), round(height * scale)))
        self.detector.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.detector.detect(small)
        if faces is None:
            return [], None, "Sin rostro"
        faces = faces.copy()
        faces[:, :14:2] *= width / small.shape[1]
        faces[:, 1:14:2] *= height / small.shape[0]
        if len(faces) != 1:
            return faces, None, "Debe aparecer una sola persona"
        face = faces[0]
        x, y, w, h = face[:4]
        if min(w, h) < 70:
            return faces, None, "Rostro demasiado pequeño para identificar"
        if x < 0 or y < 0 or x + w > width or y + h > height:
            return faces, None, "Centra el rostro completo"
        crop = self.recognizer.alignCrop(frame, face)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        if brightness < 45 or brightness > 215:
            return faces, None, "Mejora la iluminación"
        if cv2.Laplacian(gray, cv2.CV_64F).var() < 45:
            return faces, None, "Imagen borrosa: mantente quieto"
        return faces, normalize(self.recognizer.feature(crop)), "Calidad aceptable"


class StableIdentity:
    def __init__(self):
        self.reset()

    def reset(self):
        self.name = None
        self.count = 0
        self.last_time = 0.0
        self.vector = None

    def update(self, name, vector, timestamp):
        continuous = self.vector is not None and float(self.vector @ vector) >= 0.5
        if name is None:
            self.reset()
            return None
        self.count = self.count + 1 if name == self.name and timestamp - self.last_time < 0.6 and continuous else 1
        self.name, self.vector, self.last_time = name, vector, timestamp
        return name if self.count >= 3 else None


@dataclass(frozen=True)
class Frame:
    sequence: int
    captured: float
    image: np.ndarray


class Camera(threading.Thread):
    def __init__(self, source, backend, stop, width=1280, height=720, fps=30):
        super().__init__(daemon=True, name="capture")
        self.source, self.backend, self.stop = source, backend, stop
        self.width, self.height, self.requested_fps = width, height, fps
        self.latest = Latest()
        self.status = "Abriendo cámara..."
        self.fps = 0.0

    def run(self):
        cap = None
        try:
            cap = cv2.VideoCapture(self.source, self.backend)
            if not cap.isOpened():
                raise RuntimeError("No se pudo abrir la cámara. Prueba --camera 1 o --backend msmf")
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.requested_fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Best effort: backend may ignore.
            times = deque(maxlen=90)
            sequence = 0
            while not self.stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Cámara desconectada o sin imágenes. Reinicia el prototipo.")
                now = time.perf_counter()
                times.append(now)
                self.fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0
                sequence += 1
                self.latest.put(Frame(sequence, now, frame))
                self.status = f"{frame.shape[1]} × {frame.shape[0]}"
        except Exception as exc:
            self.status = str(exc)
            self.latest.put(None)
        finally:
            if cap is not None:
                cap.release()


class Worker(threading.Thread):
    def __init__(self, camera, stop, threshold=0.45, margin=0.08, hz=10):
        super().__init__(daemon=True, name="inference")
        self.camera, self.stop = camera, stop
        self.threshold, self.margin, self.hz = threshold, margin, hz
        self.latest = Latest()
        self.commands = queue.Queue(maxsize=4)
        self.status = "Cargando modelos..."
        self.names = []
        self.enrolling = False

    def run(self):
        store = None
        try:
            vision = Vision()
            store = Store(ROOT / "data/people.sqlite3")
            gallery = store.gallery()
            self.names = sorted(gallery)
            stable = StableIdentity()
            session = None
            sequence = -1
            durations = deque(maxlen=300)
            last_sample = 0.0
            self.status = "Listo"
            while not self.stop.is_set():
                start = time.perf_counter()
                try:
                    command, name = self.commands.get_nowait()
                    stable.reset()
                    if command == "enroll":
                        session = {"name": name, "vectors": [], "started": start}
                        self.enrolling = True
                        self.status = f"Registrando {name}: mira al frente y cambia ligeramente el ángulo"
                    elif command == "cancel":
                        session = None
                        self.enrolling = False
                        self.status = "Registro cancelado"
                except queue.Empty:
                    pass
                if session and start - session["started"] > 45:
                    session = None
                    self.enrolling = False
                    self.status = "Registro vencido (45 s). No se guardaron muestras parciales."
                frame = self.camera.latest.get()
                if frame is None or frame.sequence == sequence or start - frame.captured > 0.3:
                    stable.reset()
                    self.stop.wait(0.015)
                    continue
                sequence = frame.sequence
                faces, vector, quality = vision.analyze(frame.image)
                candidate, score, gap = match(vector, gallery, self.threshold, self.margin) if vector is not None else (None, 0.0, 0.0)
                identity = stable.update(candidate, vector, frame.captured) if vector is not None and not session else None
                if vector is None or session:
                    stable.reset()
                was_enrolling = session is not None
                if session and vector is not None and start - last_sample >= 0.7:
                    samples = session["vectors"]
                    if samples and float(samples[0] @ vector) < 0.5:
                        quality = "Rostro diferente: cancela y vuelve a registrar"
                    else:
                        samples.append(vector)
                        last_sample = start
                        self.status = f"Registrando {session['name']}: {len(samples)}/8 muestras"
                        if len(samples) == 8:
                            store.save(session["name"], samples)
                            self.status = f"Registrado: {session['name']}. Aléjate y vuelve para probar."
                            gallery = store.gallery()
                            self.names = sorted(gallery)
                            session = None
                            self.enrolling = False
                elapsed = (time.perf_counter() - start) * 1000
                durations.append(elapsed)
                # Publish coordinates with their exact source image, never on a newer person.
                preview = frame.image.copy()
                color = (60, 200, 100) if identity else (0, 190, 255)
                for face in faces:
                    x, y, w, h = map(int, face[:4])
                    cv2.rectangle(preview, (x, y), (x + w, y + h), color, 2)
                self.latest.put({"sequence": sequence, "captured": frame.captured, "finished": time.perf_counter(),
                    "preview": preview, "quality": quality, "identity": identity, "candidate": candidate,
                    "score": score, "gap": gap, "valid": vector is not None, "enrolling": was_enrolling,
                    "inference_ms": elapsed, "p95_ms": float(np.percentile(durations, 95)),
                    "threshold": self.threshold, "margin": self.margin})
                self.stop.wait(max(0, 1 / self.hz - (time.perf_counter() - start)))
        except Exception as exc:
            self.status = f"Error de análisis: {exc}"
            self.latest.put(None)
        finally:
            self.enrolling = False
            if store:
                store.close()
