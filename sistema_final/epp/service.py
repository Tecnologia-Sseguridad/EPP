from collections import deque
from dataclasses import dataclass, field
import threading
import time

import cv2
import numpy as np

from engine import Latest
from sistema_final.core.configuration import PROJECT_ROOT


REQUIRED = ("casco", "chaleco", "guantes")
RULES = {
    "casco": {"positive": "helmet", "negative": "no_helmet", "zone": "head"},
    "chaleco": {"positive": "vest", "negative": "none", "zone": "torso"},
    "guantes": {"positive": "gloves", "negative": "no_gloves", "zone": "hands"},
    "antiparras": {"positive": "goggles", "negative": "no_goggle", "zone": "head"},
    "botas": {"positive": "boots", "negative": "no_boots", "zone": "feet"},
}


@dataclass
class Person:
    box: list
    confidence: float
    display_id: int
    raw_box: list | None = None
    hits: int = 1
    absent: int = 0
    evidence: dict = field(default_factory=lambda: {name: deque(maxlen=15) for name in RULES})
    visible: dict = field(default_factory=lambda: {name: False for name in RULES})
    state: dict = field(default_factory=lambda: {name: "verificando" for name in RULES})
    last_explicit: dict = field(default_factory=lambda: {name: 0.0 for name in RULES})

    def __post_init__(self):
        if self.raw_box is None:
            self.raw_box = list(self.box)


def area(box):
    return max(1.0, box[2] - box[0]) * max(1.0, box[3] - box[1])


def intersection(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def remove_duplicates(detections, limit=0.65):
    selected = []
    for detection in sorted(detections, key=lambda item: item["confidence"], reverse=True):
        duplicate = any(
            detection["name"] != "Person" and detection["name"] == other["name"]
            and intersection(detection["box"], other["box"]) / min(area(detection["box"]), area(other["box"])) >= limit
            for other in selected
        )
        if not duplicate:
            selected.append(detection)
    return selected


def select_primary_person(detections, frame_shape):
    """El modo actual admite una persona: conserva el sujeto dominante y descarta ruido."""
    frame_height, frame_width = frame_shape[:2]
    frame_area = max(1.0, frame_height * frame_width)
    candidates = []
    for detection in detections:
        if detection["name"] != "Person" or detection["track_id"] is None:
            continue
        x1, y1, x2, y2 = detection["box"]
        box_width, box_height = x2 - x1, y2 - y1
        if box_width <= 0 or box_height <= 0:
            continue
        relative_area = area(detection["box"]) / frame_area
        aspect_ratio = box_width / box_height
        # Rechaza fragmentos pequeños y cajas con geometría poco compatible con una persona.
        if relative_area < 0.025 or not 0.16 <= aspect_ratio <= 1.45:
            continue
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        distance_to_center = (
            ((center_x - frame_width / 2) / frame_width) ** 2
            + ((center_y - frame_height / 2) / frame_height) ** 2
        ) ** 0.5
        score = relative_area * (1.0 - min(0.45, distance_to_center * 0.35))
        candidates.append((score, detection))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def relative_zone(epp_box, person_box):
    center_x = (epp_box[0] + epp_box[2]) / 2
    center_y = (epp_box[1] + epp_box[3]) / 2
    x1, y1, x2, y2 = person_box
    return (center_x - x1) / max(1.0, x2 - x1), (center_y - y1) / max(1.0, y2 - y1)


def compatible(epp_box, person_box, zone):
    relative_x, relative_y = relative_zone(epp_box, person_box)
    # El detector de persona puede comenzar bajo el casco o recortar levemente
    # las manos. Este margen evita perder EPP que sí pertenece al sujeto.
    if not (-0.20 <= relative_x <= 1.20 and -0.18 <= relative_y <= 1.12):
        return False
    limits = {"head": (-0.18, 0.38), "torso": (0.10, 0.80), "hands": (0.08, 0.96), "feet": (0.60, 1.12)}
    minimum, maximum = limits[zone]
    return minimum <= relative_y <= maximum


def assign_to_people(epp_detections, people):
    assigned = {track_id: {} for track_id in people}
    class_rules = {}
    for element, rule in RULES.items():
        class_rules[rule["positive"]] = (element, rule["zone"])
        class_rules[rule["negative"]] = (element, rule["zone"])
    for detection in epp_detections:
        if detection["name"] not in class_rules:
            continue
        _, zone = class_rules[detection["name"]]
        best_id, best_score = None, -10.0
        for track_id, person in people.items():
            person_box = person.raw_box or person.box
            if not compatible(detection["box"], person_box, zone):
                continue
            coverage = intersection(detection["box"], person_box) / area(detection["box"])
            relative_x, relative_y = relative_zone(detection["box"], person_box)
            distance = ((relative_x - 0.5) ** 2 + (relative_y - 0.45) ** 2) ** 0.5
            score = coverage - 0.20 * distance
            if score > best_score:
                best_id, best_score = track_id, score
        if best_id is not None and best_score >= 0.35:
            assigned[best_id].setdefault(detection["name"], []).append(detection)
    return assigned


def framing_status(box, shape):
    """Valida la distancia/encuadre antes de decidir EPP."""
    height, width = shape[:2]
    x1, y1, x2, y2 = box
    person_height = y2 - y1
    person_width = x2 - x1
    if person_height < height * 0.48:
        return "acercate"
    # Para acreditar casco, torso y guantes necesitamos margen alrededor del
    # cuerpo. Una persona que llena la cámara debe alejarse.
    if (person_height > height * 0.94 or person_width > width * 0.82
            or x1 <= 8 or x2 >= width - 8 or y1 <= 5
            or y1 + person_height * 0.90 >= height - 8):
        return "alejate"
    return "correcto"


def zone_visible(box, zone, shape):
    height, width = shape[:2]
    x1, y1, x2, y2 = box
    person_height = max(1.0, y2 - y1)
    if zone == "head":
        return y1 > 4 and y1 + 0.35 * person_height < height - 4
    if zone == "torso":
        return y1 + 0.72 * person_height < height - 4
    if zone == "hands":
        return x1 > 4 and x2 < width - 4 and y1 + 0.88 * person_height < height - 4
    return y2 < height - 4


def add_evidence(person, detections, shape):
    for element, rule in RULES.items():
        positive = detections.get(rule["positive"], [])
        negative = detections.get(rule["negative"], [])
        person.visible[element] = bool(positive or negative) or zone_visible(
            person.raw_box or person.box, rule["zone"], shape
        )
        if not person.visible[element]:
            person.evidence[element].append(None)
            continue
        # El modelo entrega una caja por guante. Una sola mano detectada no
        # acredita que la persona lleve ambos guantes.
        if element == "guantes" and len(positive) < 2:
            positive = []
        if positive and negative:
            positive_confidence = max(item["confidence"] for item in positive)
            negative_confidence = max(item["confidence"] for item in negative)
            positive, negative = (positive, []) if positive_confidence >= negative_confidence else ([], negative)
        value = 1 if positive else -1 if negative else 0
        person.evidence[element].append(value)
        if value:
            person.last_explicit[element] = time.perf_counter()


def decide(person, element):
    previous = person.state[element]
    if not person.visible[element]:
        if previous in {"si", "no"} and time.perf_counter() - person.last_explicit[element] <= 2.0:
            return previous
        return "no visible"
    values = [value for value in person.evidence[element] if value is not None]
    if not values:
        return "verificando"
    # Se conserva una conclusión ante pérdidas breves, no indefinidamente. Si
    # pasan dos segundos sin evidencia explícita, vuelve a ANALIZANDO.
    if (values[-1] == 0 and previous in {"si", "no"}
            and time.perf_counter() - person.last_explicit[element] > 2.0):
        person.state[element] = "verificando"
        return "verificando"
    # Cero significa "el modelo no concluyó", nunca "no lo lleva".
    explicit = [value for value in values[-8:] if value != 0]
    if not explicit:
        return previous
    positives = sum(value == 1 for value in explicit)
    negatives = sum(value == -1 for value in explicit)
    required = 3 if previous in {"si", "no"} else 2
    if positives >= required and positives / len(explicit) >= 0.67:
        person.state[element] = "si"
    elif negatives >= required and negatives / len(explicit) >= 0.67:
        person.state[element] = "no"
    return person.state[element]


def reconnect_distance(new_box, person):
    new_x, new_y = (new_box[0] + new_box[2]) / 2, (new_box[1] + new_box[3]) / 2
    old_x, old_y = (person.box[0] + person.box[2]) / 2, (person.box[1] + person.box[3]) / 2
    diagonal = max(1.0, ((person.box[2] - person.box[0]) ** 2 + (person.box[3] - person.box[1]) ** 2) ** 0.5)
    area_ratio = area(new_box) / area(person.box)
    if not 0.35 <= area_ratio <= 2.85:
        return None
    return ((new_x - old_x) ** 2 + (new_y - old_y) ** 2) ** 0.5 / diagonal


def _draw_equipment_icon(frame, center, element, color, state, size=38):
    """Dibuja una insignia compacta e independiente de fuentes Unicode."""
    cx, cy = center
    half = size // 2
    left, top = cx - half, cy - half
    right, bottom = cx + half, cy + half
    overlay = frame.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    cv2.rectangle(frame, (left, top), (right, bottom), (245, 245, 245), 1, cv2.LINE_AA)

    white = (250, 250, 250)
    if element == "casco":
        cv2.ellipse(frame, (cx, cy + 1), (10, 9), 180, 0, 180, white, 2, cv2.LINE_AA)
        cv2.line(frame, (cx - 13, cy + 2), (cx + 13, cy + 2), white, 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - 8), (cx, cy - 1), white, 1, cv2.LINE_AA)
    elif element == "chaleco":
        points = np.array([
            [cx - 10, cy - 11], [cx - 3, cy - 7], [cx + 3, cy - 7],
            [cx + 10, cy - 11], [cx + 12, cy + 11], [cx - 12, cy + 11],
        ], dtype=np.int32)
        cv2.polylines(frame, [points], True, white, 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - 6), (cx, cy + 11), white, 1, cv2.LINE_AA)
    else:  # guantes
        points = np.array([
            [cx - 8, cy + 10], [cx - 10, cy - 2], [cx - 7, cy - 8],
            [cx - 4, cy - 1], [cx - 2, cy - 10], [cx + 1, cy - 1],
            [cx + 4, cy - 9], [cx + 6, cy + 1], [cx + 11, cy - 2],
            [cx + 10, cy + 7], [cx + 5, cy + 12],
        ], dtype=np.int32)
        cv2.polylines(frame, [points], True, white, 2, cv2.LINE_AA)

    if state == "no":
        cv2.line(frame, (left + 5, bottom - 5), (right - 5, top + 5), white, 2, cv2.LINE_AA)
    elif state == "verificando":
        cv2.circle(frame, (right - 7, bottom - 7), 3, white, -1, cv2.LINE_AA)
    elif state == "no visible":
        cv2.putText(frame, "?", (right - 13, bottom - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, white, 1, cv2.LINE_AA)


def draw_person(frame, person, required=REQUIRED):
    x1, y1, x2, y2 = [int(round(value)) for value in person.box]
    height, width = frame.shape[:2]
    x1, x2 = max(0, x1), min(width - 1, x2)
    y1, y2 = max(0, y1), min(height - 1, y2)
    framing = framing_status(person.raw_box or person.box, frame.shape)
    if framing != "correcto":
        decisions = {element: "fuera de encuadre" for element in required}
        overall = "ALEJATE" if framing == "alejate" else "ACERCATE"
    else:
        decisions = {element: decide(person, element) for element in required}
        states = list(decisions.values())
        overall = "INCOMPLETO" if "no" in states else "COMPLETO" if all(state == "si" for state in states) else "VERIFICANDO"
    general_color = (65, 205, 85) if overall == "COMPLETO" else (45, 45, 235) if overall == "INCOMPLETO" else (40, 190, 255)
    state_colors = {
        "si": (65, 205, 85),
        "no": (45, 45, 235),
        "verificando": (40, 190, 255),
        "no visible": (150, 150, 150),
        "fuera de encuadre": (150, 150, 150),
    }
    cv2.rectangle(frame, (x1, y1), (x2, y2), general_color, 2, cv2.LINE_AA)

    # Tarjeta textual adaptable. Los pictogramas vectoriales son una guía
    # secundaria; el texto siempre comunica el resultado explícitamente.
    panel_width = 218
    header_height, row_height = 31, 29
    panel_height = header_height + len(required) * row_height + 7
    if width - x2 >= panel_width + 10:
        panel_x = x2 + 8
        panel_y = y1
    elif x1 >= panel_width + 10:
        panel_x = x1 - panel_width - 8
        panel_y = y1
    else:
        # Cuando la persona llena la imagen se usa una esquina, sin perseguir
        # el movimiento del cuerpo; esto reduce saltos visuales.
        panel_x = width - panel_width - 8
        panel_y = 8
    panel_x = min(max(6, panel_x), max(6, width - panel_width - 6))
    panel_y = min(max(6, panel_y), max(6, height - panel_height - 6))
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_width, panel_y + panel_height), (16, 18, 21), -1)
    cv2.addWeighted(overlay, 0.86, frame, 0.14, 0, frame)
    cv2.rectangle(frame, (panel_x, panel_y),
                  (panel_x + panel_width, panel_y + panel_height), (95, 99, 105), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (panel_x, panel_y),
                  (panel_x + 4, panel_y + panel_height), general_color, -1)
    cv2.putText(frame, f"EPP  {overall}", (panel_x + 14, panel_y + 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, general_color, 1, cv2.LINE_AA)

    display_states = {
        "si": "CUMPLE",
        "no": "FALTA",
        "verificando": "ANALIZANDO",
        "no visible": "NO VISIBLE",
        "fuera de encuadre": "ENCUADRA",
    }
    display_names = {
        "casco": "CASCO",
        "chaleco": "CHALECO",
        "guantes": "GUANTES",
        "antiparras": "ANTIPARRAS",
        "botas": "BOTAS",
    }
    for index, element in enumerate(required):
        state = decisions[element]
        row_y = panel_y + header_height + index * row_height
        if index:
            cv2.line(frame, (panel_x + 13, row_y), (panel_x + panel_width - 10, row_y),
                     (58, 61, 66), 1, cv2.LINE_AA)
        _draw_equipment_icon(
            frame, (panel_x + 25, row_y + row_height // 2), element,
            state_colors[state], state, size=20
        )
        cv2.putText(frame, display_names.get(element, element.upper()),
                    (panel_x + 42, row_y + 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.39, (226, 229, 232), 1, cv2.LINE_AA)
        status_text = display_states[state]
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)[0]
        cv2.putText(frame, status_text,
                    (panel_x + panel_width - text_size[0] - 10, row_y + 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, state_colors[state], 1, cv2.LINE_AA)
    return decisions, overall


class EppService(threading.Thread):
    def __init__(self, camera, stop, confidence=0.28, iou=0.45, required=REQUIRED):
        super().__init__(daemon=True, name="epp-service")
        self.camera, self.stop = camera, stop
        self.confidence, self.iou = confidence, iou
        self.required = tuple(required)
        self.enabled = threading.Event()
        self.latest = Latest()
        self.status = "En espera"

    def activate(self):
        self.enabled.set()

    def deactivate(self):
        self.enabled.clear()

    def set_required(self, required):
        self.required = tuple(required)

    def run(self):
        model = None
        people = {}
        next_person = 1
        sequence = -1
        try:
            while not self.stop.is_set():
                if not self.enabled.is_set():
                    people.clear()
                    sequence = -1
                    self.stop.wait(0.05)
                    continue
                if model is None:
                    self.status = "Cargando modelo EPP..."
                    from ultralytics import YOLO
                    model = YOLO(str(PROJECT_ROOT / "models" / "ppe_yolo26n.pt"))
                    self.status = "Listo"
                frame = self.camera.latest.get()
                if frame is None or frame.sequence == sequence or time.perf_counter() - frame.captured > 0.5:
                    self.stop.wait(0.015)
                    continue
                sequence = frame.sequence
                start = time.perf_counter()
                result = model.track(frame.image, persist=True, tracker="bytetrack.yaml", conf=self.confidence, iou=self.iou, verbose=False)[0]
                detections = []
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().tolist()
                    confidences = result.boxes.conf.cpu().tolist()
                    classes = result.boxes.cls.int().cpu().tolist()
                    ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else [None] * len(boxes)
                    for box, confidence, class_id, track_id in zip(boxes, confidences, classes, ids):
                        detections.append({"box": box, "confidence": float(confidence), "name": model.names[class_id], "track_id": track_id})
                detections = remove_duplicates(detections)
                primary_detection = select_primary_person(detections, frame.image.shape)
                for person in people.values():
                    person.absent += 1
                for detection in ([primary_detection] if primary_detection is not None else []):
                    track_id = detection["track_id"]
                    person = people.get(track_id)
                    if person is None:
                        candidates = []
                        for old_id, lost in people.items():
                            if 0 < lost.absent <= 3:
                                distance = reconnect_distance(detection["box"], lost)
                                if distance is not None and distance <= 0.65:
                                    candidates.append((distance, old_id, lost))
                        if candidates:
                            _, old_id, person = min(candidates, key=lambda item: item[0])
                            del people[old_id]
                            people[track_id] = person
                        else:
                            person = Person(detection["box"], detection["confidence"], next_person)
                            next_person += 1
                            people[track_id] = person
                    # La caja instantánea decide asociación y encuadre. La caja
                    # suavizada se usa únicamente para que el dibujo no tiemble.
                    person.raw_box = list(detection["box"])
                    # Mayor amortiguación visual: reduce el temblor sin generar demasiado retraso.
                    person.box = [old * 0.74 + new * 0.26 for old, new in zip(person.box, detection["box"])]
                    person.hits = min(100, person.hits + 1)
                    person.absent = 0
                people = {track_id: person for track_id, person in people.items() if person.absent <= 12}
                active = {track_id: person for track_id, person in people.items() if person.absent == 0}
                assigned = assign_to_people([item for item in detections if item["name"] != "Person"], active)
                preview = frame.image.copy()
                summaries = []
                for track_id, person in people.items():
                    if person.absent == 0:
                        if framing_status(person.raw_box or person.box, frame.image.shape) == "correcto":
                            add_evidence(person, assigned.get(track_id, {}), frame.image.shape)
                    if person.hits >= 2 and person.absent <= 5:
                        decisions, overall = draw_person(preview, person, self.required)
                        summaries.append({"id": person.display_id, "decisions": decisions, "overall": overall})
                annotation_mask = np.any(preview != frame.image, axis=2)
                self.latest.put({"sequence": sequence, "captured": frame.captured, "preview": preview,
                                 "annotation_mask": annotation_mask,
                                 "people": summaries, "inference_ms": (time.perf_counter() - start) * 1000})
        except Exception as exc:
            self.status = f"Error EPP: {exc}"
            self.latest.put(None)
