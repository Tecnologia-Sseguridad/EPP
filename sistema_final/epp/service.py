from dataclasses import dataclass, field
import threading
import time

import cv2
import numpy as np

from engine import Latest
from .temporal import Evidence, overall_status
from sistema_final.core.configuration import PROJECT_ROOT


REQUIRED = ("casco", "chaleco", "guantes")
NEGATIVE_MIN_CONFIDENCE = 0.40
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
    visible: dict = field(default_factory=lambda: {name: False for name in RULES})
    assessments: dict = field(default_factory=lambda: {name: Evidence() for name in RULES})
    last_seen: float = 0.0
    observations: dict = field(default_factory=dict)

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
        if detection["name"] != "Person":
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


def expanded_box(person_box, zone):
    """Amplía solo la zona lógica; no altera el seguimiento de la persona."""
    x1, y1, x2, y2 = person_box
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    if zone == "head":
        return [x1 - width * 0.18, y1 - height * 0.30,
                x2 + width * 0.18, y2]
    if zone == "hands":
        return [x1 - width * 0.32, y1 - height * 0.04,
                x2 + width * 0.32, y2 + height * 0.06]
    return [x1 - width * 0.10, y1 - height * 0.04,
            x2 + width * 0.10, y2 + height * 0.06]


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
            # Casco y manos suelen sobresalir de la caja que YOLO entrega para
            # Person. La cobertura se calcula contra una zona lógica ampliada.
            coverage = intersection(detection["box"], expanded_box(person_box, zone)) / area(detection["box"])
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
    # A esta escala el modelo todavía conserva detalle suficiente. El límite
    # anterior (48 %) obligaba innecesariamente a mostrar casi todo el cuerpo.
    if person_height < height * 0.38:
        return "acercate"
    # Para acreditar casco, torso y guantes necesitamos margen alrededor del
    # cuerpo. Una persona que llena la cámara debe alejarse.
    if (person_height > height * 0.985 or person_width > width * 0.90
            or x1 <= 3 or x2 >= width - 3 or y1 <= 2
            or y1 + person_height * 0.97 >= height - 3):
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


def distinct_pair(detections):
    """Two distinct boxes; one glove moving across the torso is never a pair."""
    for index, first in enumerate(detections):
        for second in detections[index + 1:]:
            overlap = intersection(first["box"], second["box"])
            if overlap / min(area(first["box"]), area(second["box"])) < 0.15:
                return True
    return False


def add_evidence(person, detections, shape, now=None):
    now = time.perf_counter() if now is None else now
    for element, rule in RULES.items():
        positive = [item for item in detections.get(rule["positive"], [])
                    if item["confidence"] >= 0.35]
        negative = [item for item in detections.get(rule["negative"], [])
                    if item["confidence"] >= NEGATIVE_MIN_CONFIDENCE]
        person.visible[element] = bool(positive or negative) or zone_visible(
            person.raw_box or person.box, rule["zone"], shape)
        if element == "guantes":
            # A bare hand contradicts compliance even with a glove on the other.
            value = -1 if negative else 1 if distinct_pair(positive) else 0
        elif positive and negative:
            value = None  # Conflicting observations cannot establish compliance.
        else:
            value = 1 if positive else -1 if negative else 0
        person.assessments[element].observe(value, now)
        person.observations[element] = {
            "positive": len(positive), "negative": len(negative),
            "confidence": max([d["confidence"] for d in positive + negative], default=0),
        }


def decide(person, element, now=None):
    now = time.perf_counter() if now is None else now
    state = person.assessments[element].decision(now)
    return state if state in {"si", "no"} else (
        "verificando" if person.visible[element] else "no visible")


def reconnect_distance(new_box, person):
    new_x, new_y = (new_box[0] + new_box[2]) / 2, (new_box[1] + new_box[3]) / 2
    old_x, old_y = (person.box[0] + person.box[2]) / 2, (person.box[1] + person.box[3]) / 2
    diagonal = max(1.0, ((person.box[2] - person.box[0]) ** 2 + (person.box[3] - person.box[1]) ** 2) ** 0.5)
    area_ratio = area(new_box) / area(person.box)
    # El casco, los brazos y una inclinación pueden cambiar mucho la caja de
    # Person sin que haya cambiado el sujeto.
    if not 0.18 <= area_ratio <= 5.5:
        return None
    return ((new_x - old_x) ** 2 + (new_y - old_y) ** 2) ** 0.5 / diagonal


def draw_person(frame, person, required=REQUIRED):
    # Unobtrusive monochrome outline; results belong outside the image.
    x1, y1, x2, y2 = person.box
    w, h = x2 - x1, y2 - y1
    height, width = frame.shape[:2]
    p1 = (max(0, round(x1 - w * .06)), max(0, round(y1 - h * .10)))
    p2 = (min(width - 1, round(x2 + w * .06)), min(height - 1, round(y2)))
    cv2.rectangle(frame, p1, p2, (25, 25, 25), 3, cv2.LINE_AA)
    cv2.rectangle(frame, p1, p2, (235, 235, 235), 1, cv2.LINE_AA)
    decisions = {name: decide(person, name) for name in required}
    return decisions, overall_status(decisions)


class Presence:
    """Single-person spatial continuity; not an identity recognition mechanism."""
    def __init__(self):
        self.person = None
        self.next_id = 1

    def reset(self):
        self.person = None

    def update(self, detection, now):
        previous = self.person
        if detection is None:
            if previous and now - previous.last_seen > .65:
                self.reset()
            return None
        keep = False
        if previous and now - previous.last_seen <= .65:
            box = detection["box"]
            distance = reconnect_distance(box, previous)
            overlap = intersection(box, previous.raw_box) / max(1, min(area(box), area(previous.raw_box)))
            keep = distance is not None and distance < .30 and overlap > .30
        if not keep:
            self.person = Person(list(detection["box"]), detection["confidence"], self.next_id)
            self.next_id += 1
        person = self.person
        person.raw_box = list(detection["box"])
        person.box = [old * .65 + new * .35 for old, new in zip(person.box, person.raw_box)]
        person.hits += 1
        person.last_seen = now
        return person


class EppService(threading.Thread):
    def __init__(self, camera, stop, confidence=0.28, iou=0.45, required=REQUIRED):
        super().__init__(daemon=True, name="epp-service")
        self.camera, self.stop = camera, stop
        self.confidence, self.iou = confidence, iou
        self.required = tuple(required)
        self.enabled = threading.Event()
        self.latest = Latest()
        self.status = "En espera"
        self.reset_requested = threading.Event()

    def activate(self):
        self.enabled.set()

    def deactivate(self):
        self.enabled.clear()
        self.latest.put(None)
        self.reset_requested.set()

    def set_required(self, required):
        self.required = tuple(required)
        self.latest.put(None)
        self.reset_requested.set()

    def run(self):
        model = None
        presence = Presence()
        sequence = -1
        try:
            while not self.stop.is_set():
                if self.reset_requested.is_set():
                    presence.reset()
                    self.reset_requested.clear()
                if not self.enabled.is_set():
                    presence.reset()
                    self.latest.put(None)
                    self.stop.wait(.05)
                    continue
                if model is None:
                    self.status = "Cargando detector..."
                    from ultralytics import YOLO
                    model = YOLO(str(PROJECT_ROOT / "models" / "ppe_yolo26n.pt"))
                    self.status = "Listo"
                frame = self.camera.latest.get()
                if frame is None or frame.sequence == sequence or time.perf_counter() - frame.captured > .5:
                    self.stop.wait(.015)
                    continue
                sequence = frame.sequence
                start = time.perf_counter()
                # Equipment predictions must not wait for tracking confirmation.
                result = model.predict(frame.image, imgsz=640, conf=self.confidence,
                                       iou=self.iou, verbose=False)[0]
                detections = []
                if result.boxes is not None:
                    for box, confidence, class_id in zip(
                            result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
                            result.boxes.cls.int().cpu().tolist()):
                        detections.append({"box": box, "confidence": float(confidence),
                                           "name": model.names[class_id], "track_id": None})
                detections = remove_duplicates(detections)
                primary = select_primary_person(detections, frame.image.shape)
                others = [item for item in detections if item["name"] == "Person"
                          and primary is not None and item is not primary
                          and area(item["box"]) > .55 * area(primary["box"])]
                ambiguous = bool(others)
                if ambiguous:
                    presence.reset()
                person = presence.update(None if ambiguous else primary, frame.captured)
                preview = frame.image.copy()
                summaries = []
                equipment = []
                guidance = "Debe aparecer una sola persona" if ambiguous else "Ubíquese frente a la cámara"
                if person is not None:
                    assigned = assign_to_people(
                        [item for item in detections if item["name"] != "Person"], {person.display_id: person})
                    equipment = [item for items in assigned[person.display_id].values() for item in items]
                    # Evaluate each item, even when legs touch the image border.
                    add_evidence(person, assigned[person.display_id], frame.image.shape)
                    decisions, overall = draw_person(preview, person, self.required)
                    framing = framing_status(person.raw_box, frame.image.shape)
                    guidance = ("Acérquese: falta detalle en la imagen" if framing == "acercate"
                                else "Deje visibles la cabeza, el torso y ambas manos")
                    if framing == "acercate":
                        decisions = {name: "verificando" for name in self.required}
                        overall = "VERIFICANDO"
                    summaries.append({"id": person.display_id, "decisions": decisions,
                                      "overall": overall, "observations": person.observations,
                                      "framing": framing})
                elapsed = (time.perf_counter() - start) * 1000
                self.latest.put({"sequence": sequence, "captured": frame.captured,
                                 "preview": preview, "annotation_mask": np.any(preview != frame.image, axis=2),
                                 "people": summaries, "equipment": equipment,
                                 "guidance": guidance, "inference_ms": elapsed})
        except Exception as exc:
            self.status = f"Error EPP: {exc}"
            self.latest.put(None)
