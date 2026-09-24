"""Exercise real model inference without displaying or retaining camera images."""
import argparse
import json
import platform
import time
import threading

import cv2
import numpy as np

from engine import Camera, Vision, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seconds", type=int, default=15)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--backend", choices=["dshow", "msmf", "auto"], default="dshow")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    vision = Vision()
    durations = []
    valid = faces_seen = 0
    stop = threading.Event()
    camera = None
    if not args.synthetic:
        backend = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "auto": cv2.CAP_ANY}[args.backend]
        camera = Camera(args.camera, backend, stop, args.width, args.height)
        camera.start()
    deadline = time.perf_counter() + args.seconds
    last = -1
    try:
        while time.perf_counter() < deadline:
            if camera:
                frame = camera.latest.get()
                if frame is None or frame.sequence == last:
                    if not camera.is_alive():
                        break
                    time.sleep(0.01)
                    continue
                image = frame.image
                last = frame.sequence
            else:
                image = np.zeros((720, 1280, 3), np.uint8)
            started = time.perf_counter()
            faces, vector, _ = vision.analyze(image)
            durations.append((time.perf_counter() - started) * 1000)
            faces_seen += int(len(faces) > 0)
            valid += int(vector is not None)
    finally:
        stop.set()
        if camera:
            camera.join(timeout=2)
    report = {"mode": "synthetic_no_face" if args.synthetic else "camera", "python": platform.python_version(),
              "backend": args.backend, "requested_resolution": [args.width, args.height],
              "opencv": cv2.__version__, "samples": len(durations), "frames_with_faces": faces_seen,
              "valid_embeddings": valid, "camera_status": camera.status if camera else None,
              "capture_fps": camera.fps if camera else None,
              "median_ms": float(np.median(durations)) if durations else None,
              "p95_ms": float(np.percentile(durations, 95)) if durations else None,
              "note": "Sin rostros válidos no mide reconocimiento. No mide precisión ni latencia física de pantalla."}
    folder = ROOT / "data"
    folder.mkdir(exist_ok=True)
    (folder / f"benchmark_{report['mode']}_{args.backend}_{args.width}x{args.height}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not durations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
