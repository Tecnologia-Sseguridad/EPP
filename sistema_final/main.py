import argparse
from pathlib import Path
import sys
import threading
import tkinter as tk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from sistema_final.core.camera import Camera
from sistema_final.core.configuration import load_config
from sistema_final.core.database import get_required_epp, initialize_database
from sistema_final.epp.service import EppService
from sistema_final.facial.service import FacialService
from sistema_final.ui.main_window import MainWindow
from sistema_final.voices import VoiceAssistant


def main():
    parser = argparse.ArgumentParser(description="Sistema integrado facial y EPP")
    parser.add_argument("--smoke-test", action="store_true")
    arguments = parser.parse_args()
    config = load_config()
    initialize_database()
    stop = threading.Event()
    backends = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "auto": cv2.CAP_ANY}
    camera_value = str(config["camera"])
    source = int(camera_value) if camera_value.isdigit() else camera_value
    camera = Camera(source, backends[config["backend"]], stop, config["width"], config["height"], config["fps"])
    facial = FacialService(
        camera, stop, config["facial_threshold"], config["facial_margin"],
        config["facial_hz"], config.get("anti_spoofing_enabled", True),
        config.get("liveness_real_threshold", 0.70),
        config.get("liveness_fake_threshold", 0.20),
    )
    epp = EppService(
        camera, stop, config["epp_confidence"], config["epp_iou"],
        required=get_required_epp(),
    )
    voice = VoiceAssistant(stop)
    camera.start()
    facial.start()
    epp.start()
    voice.start()
    root = tk.Tk()
    app = MainWindow(root, camera, facial, epp, stop, config, voice)
    if arguments.smoke_test:
        root.after(5000, app.close)
    root.mainloop()
    stop.set()
    facial.join(timeout=2)
    epp.join(timeout=2)
    voice.join(timeout=2)
    camera.join(timeout=2)


if __name__ == "__main__":
    main()
