from collections import deque
import queue
import threading
import time

import cv2
import numpy as np

from engine import Latest, StableIdentity, Store, Vision, match
from sistema_final.core.configuration import PROJECT_ROOT
from sistema_final.core.database import DATABASE_PATH, save_enrollment_images, sync_people
from sistema_final.facial.anti_spoofing import AntiSpoofingEngine


class FacialService(threading.Thread):
    """Adaptador activable que conserva la lógica facial del prototipo."""

    def __init__(self, camera, stop, threshold=0.45, margin=0.08, hz=10,
                 anti_spoofing=True, liveness_real_threshold=0.70,
                 liveness_fake_threshold=0.20):
        super().__init__(daemon=True, name="facial-service")
        self.camera = camera
        self.stop = stop
        self.threshold = threshold
        self.margin = margin
        self.hz = hz
        self.anti_spoofing = anti_spoofing
        self.liveness_real_threshold = liveness_real_threshold
        self.liveness_fake_threshold = liveness_fake_threshold
        self.enabled = threading.Event()
        self.latest = Latest()
        self.commands = queue.Queue(maxsize=8)
        self.status = "En espera"
        self.names = []
        self.enrolling = False

    def activate(self):
        self.enabled.set()

    def deactivate(self):
        if not self.enrolling:
            self.enabled.clear()

    def command(self, action, name=""):
        self.commands.put_nowait((action, name))
        self.enabled.set()

    def run(self):
        store = None
        try:
            vision = Vision()
            liveness_engine = None
            if self.anti_spoofing:
                self.status = "Cargando prueba de vida..."
                liveness_engine = AntiSpoofingEngine(
                    PROJECT_ROOT / "models",
                    real_threshold=self.liveness_real_threshold,
                    fake_threshold=self.liveness_fake_threshold,
                )
            store = Store(DATABASE_PATH)
            gallery = store.gallery()
            self.names = sorted(gallery)
            stable = StableIdentity()
            previous_vector = None
            previous_capture = 0.0
            session = None
            sequence = -1
            durations = deque(maxlen=300)
            last_sample = 0.0
            self.status = "Listo"
            while not self.stop.is_set():
                try:
                    action, name = self.commands.get_nowait()
                    stable.reset()
                    if liveness_engine:
                        liveness_engine.reset()
                    if action == "enroll":
                        session = {"name": name, "vectors": [], "previews": [], "started": time.perf_counter()}
                        self.enrolling = True
                        self.status = f"Registrando {name}: mira al frente y cambia ligeramente el ángulo"
                    elif action == "cancel":
                        session = None
                        self.enrolling = False
                        self.status = "Registro cancelado"
                    elif action == "reload":
                        gallery = store.gallery()
                        self.names = sorted(gallery)
                except queue.Empty:
                    pass
                if not self.enabled.is_set() and session is None:
                    stable.reset()
                    if liveness_engine:
                        liveness_engine.reset()
                    self.stop.wait(0.05)
                    continue
                start = time.perf_counter()
                if session and start - session["started"] > 45:
                    session = None
                    self.enrolling = False
                    self.status = "Registro vencido (45 s). No se guardaron muestras parciales."
                frame = self.camera.latest.get()
                if frame is not None and frame.sequence == sequence and start - frame.captured <= 0.3:
                    self.stop.wait(0.015)
                    continue
                if frame is None or start - frame.captured > 0.3:
                    stable.reset()
                    if liveness_engine:
                        liveness_engine.reset()
                    self.stop.wait(0.015)
                    continue
                sequence = frame.sequence
                faces, vector, quality = vision.analyze(frame.image)
                # Liveness history belongs to one continuous face observation.
                if vector is not None and (
                        previous_vector is None or frame.captured - previous_capture > .5
                        or float(previous_vector @ vector) < .55):
                    stable.reset()
                    if liveness_engine:
                        liveness_engine.reset()
                previous_vector = vector.copy() if vector is not None else None
                previous_capture = frame.captured
                face_size = min(faces[0][2], faces[0][3]) if vector is not None and len(faces) == 1 else 0.0
                # Entre 70 y 99 px permitimos reconocer desde más lejos, pero
                # exigimos mayor similitud y separación para no aumentar las
                # falsas aceptaciones. Desde 100 px conserva la calibración original.
                distance_penalty = 0.05 if face_size < 100 else 0.0
                margin_penalty = 0.02 if face_size < 100 else 0.0
                candidate, score, gap = (
                    match(vector, gallery, self.threshold + distance_penalty,
                          self.margin + margin_penalty)
                    if vector is not None else (None, 0.0, 0.0)
                )
                liveness, liveness_score, liveness_raw = "desactivado", 0.0, 0.0
                if vector is not None and len(faces) == 1 and liveness_engine:
                    liveness, liveness_score, liveness_raw = liveness_engine.analyze(
                        frame.image, faces[0]
                    )
                elif vector is None and liveness_engine:
                    liveness_engine.reset()
                    liveness = "sin rostro"
                elif vector is not None:
                    quality = "Prueba de vida desactivada: no se confirma identidad"
                identity = (
                    stable.update(candidate, vector, frame.captured)
                    if vector is not None and not session and liveness == "real"
                    else None
                )
                if vector is None or session or liveness != "real":
                    stable.reset()
                was_enrolling = session is not None
                if session and vector is not None and face_size < 100:
                    quality = "Acércate para registrar plantillas de buena calidad"
                if (session and vector is not None and face_size >= 100
                        and liveness == "real" and start - last_sample >= 0.7):
                    samples = session["vectors"]
                    if samples and float(samples[0] @ vector) < 0.5:
                        quality = "Rostro diferente: cancela y vuelve a registrar"
                    else:
                        samples.append(vector)
                        session["previews"].append(vision.recognizer.alignCrop(frame.image, faces[0]).copy())
                        last_sample = start
                        self.status = f"Registrando {session['name']}: {len(samples)}/8 muestras"
                        if len(samples) == 8:
                            store.save(session["name"], samples)
                            save_enrollment_images(session["name"],session["previews"])
                            gallery = store.gallery()
                            self.names = sorted(gallery)
                            sync_people()
                            session = None
                            self.enrolling = False
                            self.status = "Registro guardado correctamente"
                elapsed = (time.perf_counter() - start) * 1000
                durations.append(elapsed)
                preview = frame.image.copy()
                color = (235, 235, 235)
                for face in faces:
                    x, y, width, height = map(int, face[:4])
                    cv2.rectangle(preview, (x, y), (x + width, y + height), (25, 25, 25), 3)
                    cv2.rectangle(preview, (x, y), (x + width, y + height), color, 1)
                annotation_mask = np.any(preview != frame.image, axis=2)
                self.latest.put({
                    "sequence": sequence, "captured": frame.captured, "preview": preview,
                    "annotation_mask": annotation_mask,
                    "quality": quality, "identity": identity, "candidate": candidate,
                    "score": score, "gap": gap, "valid": vector is not None,
                    "face_size": face_size,
                    "liveness": liveness, "liveness_score": liveness_score,
                    "liveness_raw": liveness_raw,
                    "enrolling": was_enrolling, "inference_ms": elapsed,
                    "p95_ms": float(np.percentile(durations, 95)),
                })
                self.stop.wait(max(0, 1 / self.hz - (time.perf_counter() - start)))
        except Exception as exc:
            self.status = f"Error facial: {exc}"
            self.latest.put(None)
        finally:
            self.enrolling = False
            if store:
                store.close()
