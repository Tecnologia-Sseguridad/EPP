"""Lector experimental de documentos TD1/MRZ con cámara en vivo.

Uso: venv\Scripts\python.exe probar_mrz.py
No guarda imágenes ni datos personales.
"""
from dataclasses import dataclass
from datetime import datetime
import queue
import re
import threading
import time

import cv2
import numpy as np
from mrzscanner import MRZScanner, ModelType


ALLOWED = re.compile(r"[^A-Z0-9<]")
WEIGHTS = (7, 3, 1)


def checksum(value):
    def number(character):
        if character.isdigit():
            return int(character)
        if "A" <= character <= "Z":
            return ord(character) - 55
        return 0
    return str(sum(number(char) * WEIGHTS[index % 3] for index, char in enumerate(value)) % 10)


def numeric(value):
    return value.translate(str.maketrans({"O": "0", "Q": "0", "I": "1", "L": "1",
                                          "Z": "2", "S": "5", "G": "6", "B": "8"}))


def readable_date(value, birth=False):
    value = numeric(value)
    if len(value) != 6 or not value.isdigit():
        return value
    year, month, day = int(value[:2]), int(value[2:4]), int(value[4:6])
    current = datetime.now().year % 100
    full_year = (1900 if birth and year > current else 2000) + year
    try:
        return datetime(full_year, month, day).strftime("%d-%m-%Y")
    except ValueError:
        return value


def chile_rut(optional):
    match = re.search(r"(\d{7,8})<([0-9K])<", numeric(optional))
    if not match:
        return ""
    body, verifier = match.groups()
    total, multiplier = 0, 2
    for digit in reversed(body):
        total += int(digit) * multiplier
        multiplier = 2 if multiplier == 7 else multiplier + 1
    expected_value = 11 - total % 11
    expected = "0" if expected_value == 11 else "K" if expected_value == 10 else str(expected_value)
    if verifier != expected:
        return ""
    grouped = f"{int(body):,}".replace(",", ".")
    return f"{grouped}-{verifier}"


def parse_td1(lines):
    if len(lines) != 3 or any(len(line) != 30 for line in lines):
        return None
    first, second, third = lines
    issuer = first[2:5]
    document_field = numeric(first[5:14]) if issuer == "CHL" else first[5:14]
    birth, expiry = numeric(second[:6]), numeric(second[8:14])
    checks = {
        "documento": checksum(document_field) == numeric(first[14]),
        "nacimiento": checksum(birth) == numeric(second[6]),
        "vencimiento": checksum(expiry) == numeric(second[14]),
    }
    composite = document_field + first[14:30] + birth + numeric(second[6]) + expiry + numeric(second[14]) + second[18:29]
    checks["compuesto"] = checksum(composite) == numeric(second[29])
    if sum(checks.values()) < 3 or not checks["compuesto"]:
        return None
    names = [part.replace("<", " ").strip() for part in third.split("<<", 1)]
    surname = re.sub(r"\s+", " ", names[0]).title()
    given = re.sub(r"\s+", " ", names[1] if len(names) > 1 else "").title()
    return {
        "nombre": (given + " " + surname).strip(),
        "documento": first[5:14].replace("<", ""),
        "rut": chile_rut(second[18:29]) if issuer == "CHL" else "",
        "nacionalidad": second[15:18].replace("<", ""),
        "nacimiento": readable_date(birth, birth=True),
        "vencimiento": readable_date(expiry),
        "sexo": second[7].replace("<", "—"),
        "checks": checks,
        "raw": lines,
    }


def normalize_output(raw_lines):
    candidates = []
    for raw in raw_lines:
        line = ALLOWED.sub("", raw.upper().replace(" ", ""))
        if len(line) == 29 and line.startswith("<"):
            line = "I" + line
        if 29 <= len(line) <= 32:
            candidates.append(line[:30].ljust(30, "<"))
    if len(candidates) != 3:
        return []
    # Corrige únicamente posiciones cuyo alfabeto está definido por ICAO.
    letter_fix = str.maketrans({"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B"})
    first, second, third = candidates
    first = first[:5].translate(letter_fix) + first[5:]
    second = second[:15] + second[15:18].translate(letter_fix) + second[18:]
    return [first, second, third]


def rectify_card(region):
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 55, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        if cv2.contourArea(contour) < region.shape[0] * region.shape[1] * 0.35:
            continue
        polygon = cv2.approxPolyDP(contour, 0.025 * cv2.arcLength(contour, True), True)
        if len(polygon) != 4:
            continue
        points = polygon.reshape(4, 2).astype(np.float32)
        sums, diffs = points.sum(axis=1), np.diff(points, axis=1).ravel()
        ordered = np.array([points[np.argmin(sums)], points[np.argmin(diffs)],
                            points[np.argmax(sums)], points[np.argmax(diffs)]], dtype=np.float32)
        target = np.array([[0, 0], [855, 0], [855, 539], [0, 539]], dtype=np.float32)
        return cv2.warpPerspective(region, cv2.getPerspectiveTransform(ordered, target), (856, 540))
    return cv2.resize(region, (856, 540), interpolation=cv2.INTER_CUBIC)


def prepare_mrz(card):
    # El modelo especializado reconoce el bloque completo y su alfabeto OCR-B,
    # incluido el separador '<'. Se conserva color para no destruir trazos finos.
    return card[int(card.shape[0] * 0.62):int(card.shape[0] * 0.96), :]


@dataclass
class OcrResult:
    data: dict | None
    elapsed_ms: float
    lines: list
    raw_lines: list


class MrzWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.input = queue.Queue(maxsize=1)
        self.latest = None
        self.running = True
        self.busy = False

    def submit(self, image):
        if self.busy:
            return
        try:
            self.input.put_nowait(image.copy())
        except queue.Full:
            pass

    def run(self):
        engine = MRZScanner(
            model_type=ModelType.recognition,
            recognition_cfg="20250221",
            session_option={"intra_op_num_threads": 2, "inter_op_num_threads": 1},
        )
        while self.running:
            try:
                image = self.input.get(timeout=0.1)
            except queue.Empty:
                continue
            self.busy = True
            started = time.perf_counter()
            try:
                card = rectify_card(image)
                output = engine(prepare_mrz(card), do_center_crop=False, do_postprocess=False)
                raw_lines = output.get("mrz_texts") or []
                lines = normalize_output(raw_lines)
                self.latest = OcrResult(parse_td1(lines), (time.perf_counter() - started) * 1000,
                                        lines, list(raw_lines))
            except Exception:
                self.latest = OcrResult(None, (time.perf_counter() - started) * 1000, [], [])
            finally:
                self.busy = False


def corners(frame, x1, y1, x2, y2, color):
    length, thickness = 34, 3
    for x, direction in ((x1, 1), (x2, -1)):
        for y, vertical in ((y1, 1), (y2, -1)):
            cv2.line(frame, (x, y), (x + direction * length, y), color, thickness, cv2.LINE_AA)
            cv2.line(frame, (x, y), (x, y + vertical * length), color, thickness, cv2.LINE_AA)


def main():
    cv2.setNumThreads(1)
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    if not camera.isOpened():
        raise SystemExit("No se pudo abrir la cámara.")
    worker = MrzWorker(); worker.start()
    accepted, repeated, invalid_count, last_result, last_submit = None, 0, 0, None, 0.0
    frame_times = []
    title = "Lector MRZ de visitantes"
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            frame_times.append(time.perf_counter())
            frame_times = frame_times[-60:]
            camera_fps = ((len(frame_times) - 1) / (frame_times[-1] - frame_times[0])) if len(frame_times) > 1 else 0
            guide_width = min(int(width * 0.76), int(height * 0.72 * 1.586))
            guide_height = int(guide_width / 1.586)
            x1, y1 = (width - guide_width) // 2, max(70, (height - guide_height) // 2 - 20)
            x2, y2 = x1 + guide_width, y1 + guide_height
            region = frame[y1:y2, x1:x2]
            now = time.perf_counter()
            if accepted is None and now - last_submit >= 1.20 and not worker.busy:
                worker.submit(region)
                last_submit = now
            result = worker.latest
            if result is not None and result is not last_result:
                last_result = result
                if result.data:
                    raw_key = tuple(result.data["raw"])
                    if accepted and tuple(accepted["raw"]) == raw_key:
                        repeated += 1
                    else:
                        accepted, repeated = result.data, 1
                    invalid_count = 0
                else:
                    invalid_count += 1
                    if invalid_count >= 2:
                        accepted, repeated = None, 0
            valid = accepted is not None and repeated >= 1
            color = (65, 205, 85) if valid else (40, 190, 255)
            corners(frame, x1, y1, x2, y2, color)
            mrz_y = y1 + int(guide_height * .62)
            cv2.line(frame, (x1 + 12, mrz_y), (x2 - 12, mrz_y), color, 1, cv2.LINE_AA)
            cv2.putText(frame, "UBIQUE EL REVERSO DE LA CEDULA DENTRO DE LA GUIA", (x1, y1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, .57, (245, 245, 245), 1, cv2.LINE_AA)
            cv2.putText(frame, "MRZ hacia abajo  |  Q o ESC para salir", (x1, y2 + 27),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, (205, 210, 215), 1, cv2.LINE_AA)
            status = "DOCUMENTO LEIDO" if valid else "LEYENDO MRZ..."
            cv2.putText(frame, status, (24, 38), cv2.FONT_HERSHEY_SIMPLEX, .72, color, 2, cv2.LINE_AA)
            if result:
                cv2.putText(frame, f"CAM {camera_fps:.0f} FPS  |  OCR {result.elapsed_ms:.0f} ms", (width - 260, 34),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, (205, 210, 215), 1, cv2.LINE_AA)
                if not valid and result.raw_lines:
                    debug_y = y2 - 64
                    for index, raw_line in enumerate(result.raw_lines[:3]):
                        cv2.putText(frame, raw_line[:34], (x1 + 15, debug_y + index * 18),
                                    cv2.FONT_HERSHEY_SIMPLEX, .40, (225, 225, 225), 1, cv2.LINE_AA)
            if valid:
                panel_y = height - 112
                overlay = frame.copy(); cv2.rectangle(overlay, (15, panel_y), (width - 15, height - 12), (15, 18, 22), -1)
                frame = cv2.addWeighted(overlay, .88, frame, .12, 0)
                fields = [("NOMBRE", accepted["nombre"]), ("RUT", accepted.get("rut") or "—"),
                          ("DOCUMENTO", accepted["documento"]),
                          ("NACIONALIDAD", accepted["nacionalidad"]), ("NACIMIENTO", accepted["nacimiento"]),
                          ("VENCIMIENTO", accepted["vencimiento"])]
                column = (width - 50) // len(fields)
                for index, (label, value) in enumerate(fields):
                    left = 28 + index * column
                    cv2.putText(frame, label, (left, panel_y + 29), cv2.FONT_HERSHEY_SIMPLEX, .38, (145, 150, 158), 1, cv2.LINE_AA)
                    cv2.putText(frame, value[:24], (left, panel_y + 61), cv2.FONT_HERSHEY_SIMPLEX, .52, (245, 245, 245), 1, cv2.LINE_AA)
            cv2.imshow(title, frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
                break
    finally:
        worker.running = False
        worker.join(timeout=1)
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
