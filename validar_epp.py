"""Evaluate labelled local video clips through the EPP decision pipeline."""
import argparse
import json
from pathlib import Path

import cv2

from sistema_final.core.configuration import PROJECT_ROOT, load_config
from sistema_final.epp.service import (
    Presence, add_evidence, area, assign_to_people, decide,
    framing_status, remove_duplicates, select_primary_person,
)


def summarize(observations):
    summary = {}
    for item in sorted({row["item"] for row in observations}):
        rows = [r for r in observations if r["item"] == item]
        tp = sum(r["expected"] == "si" and r["predicted"] == "si" for r in rows)
        fp = sum(r["expected"] == "no" and r["predicted"] == "si" for r in rows)
        tn = sum(r["expected"] == "no" and r["predicted"] == "no" for r in rows)
        fn = sum(r["expected"] == "si" and r["predicted"] == "no" for r in rows)
        unknown = len(rows) - tp - fp - tn - fn
        summary[item] = {
            "observaciones": len(rows), "correctas": tp + tn,
            "falsos_cumplimientos": fp, "falsas_faltas": fn,
            "sin_conclusion": unknown,
            "precision_del_cumplimiento": tp / (tp + fp) if tp + fp else None,
            "cobertura": (len(rows) - unknown) / len(rows) if rows else None,
            "acierto_incluyendo_abstenciones": (tp + tn) / len(rows) if rows else None,
        }
    return summary


def evaluate(manifest_path, hz=5):
    from ultralytics import YOLO
    manifest_path = Path(manifest_path).resolve()
    clips = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(clips, list) or not clips:
        raise ValueError("El manifiesto debe ser una lista de vídeos etiquetados.")
    config = load_config()
    model = YOLO(str(PROJECT_ROOT / "models" / "ppe_yolo26n.pt"))
    observations = []
    for clip_index, clip in enumerate(clips):
        expected = clip["expected"]
        allowed = {"casco", "chaleco", "guantes", "antiparras", "botas"}
        if not expected or not set(expected) <= allowed or any(v not in ("si", "no") for v in expected.values()):
            raise ValueError("Cada expected debe indicar EPP y valores si/no.")
        video = (manifest_path.parent / clip["video"]).resolve()
        cap = cv2.VideoCapture(str(video))
        session = Presence()
        frames, next_sample = 0, 0.
        count_before = len(observations)
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not cap.isOpened() or fps <= 0:
                raise ValueError(f"No se pudo abrir el vídeo: {video.name}")
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp = frames / fps
                frames += 1
                if timestamp < next_sample:
                    continue
                next_sample = timestamp + 1 / hz
                prediction = model.predict(frame, imgsz=640, conf=config["epp_confidence"],
                                           iou=config["epp_iou"], verbose=False)[0]
                boxes = prediction.boxes
                detections = [] if boxes is None else [
                    {"box": box, "confidence": confidence, "name": model.names[class_id], "track_id": None}
                    for box, confidence, class_id in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(),
                                                        boxes.cls.int().cpu().tolist())]
                detections = remove_duplicates(detections)
                primary = select_primary_person(detections, frame.shape)
                ambiguous = any(d["name"] == "Person" and d is not primary and primary is not None
                                and area(d["box"]) > .55 * area(primary["box"]) for d in detections)
                if ambiguous:
                    session.reset()
                person = session.update(None if ambiguous else primary, timestamp)
                if person:
                    assigned = assign_to_people(detections, {person.display_id: person})
                    add_evidence(person, assigned[person.display_id], frame.shape, now=timestamp)
                for item, truth in expected.items():
                    result = decide(person, item, now=timestamp) if person else "verificando"
                    if person and framing_status(person.raw_box, frame.shape) == "acercate":
                        result = "verificando"
                    observations.append({"clip": clip_index + 1, "seconds": round(timestamp, 3),
                                         "item": item, "expected": truth, "predicted": result})
        finally:
            cap.release()
        if len(observations) == count_before:
            raise ValueError(f"Vídeo sin fotogramas utilizables: {video.name}")
    return {"nota": "Métricas por observación; los fotogramas de un vídeo no son ensayos independientes.",
            "clips": len(clips), "summary": summarize(observations), "observations": observations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="JSON con video y expected por clip")
    parser.add_argument("--output", default="evaluacion_epp.json")
    parser.add_argument("--hz", type=float, default=5)
    args = parser.parse_args()
    if not 1 <= args.hz <= 30:
        parser.error("--hz debe estar entre 1 y 30")
    report = evaluate(args.manifest, args.hz)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
