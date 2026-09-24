import argparse
from pathlib import Path
import sys
import threading

PROJECT_ROOT=Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
import cv2
from sistema_final.core.camera import Camera
from sistema_final.core.configuration import load_config
from sistema_final.core.database import get_required_epp,initialize_database
from sistema_final.epp.service import EppService
from sistema_final.facial.service import FacialService
from sistema_final.ui.qt_epp_totem import run_epp_totem
from sistema_final.voices import VoiceAssistant


def main():
    parser=argparse.ArgumentParser(description="Modo tótem de verificación EPP");parser.add_argument("--smoke-test",action="store_true");args=parser.parse_args();config=load_config();initialize_database();stop=threading.Event();backend={"dshow":cv2.CAP_DSHOW,"msmf":cv2.CAP_MSMF,"auto":cv2.CAP_ANY}[config["backend"]];value=str(config["camera"]);source=int(value) if value.isdigit() else value
    camera=Camera(source,backend,stop,config["width"],config["height"],config["fps"])
    facial=FacialService(camera,stop,config["facial_threshold"],config["facial_margin"],config["facial_hz"],config.get("anti_spoofing_enabled",True),config.get("liveness_real_threshold",.7),config.get("liveness_fake_threshold",.2))
    epp=EppService(camera,stop,config["epp_confidence"],config["epp_iou"],get_required_epp())
    voice=VoiceAssistant(stop,enabled=config.get("voice_enabled",True));camera.start();facial.start();epp.start();voice.start()
    try:run_epp_totem(camera,epp,facial,stop,config,voice,args.smoke_test)
    finally:stop.set();facial.join(timeout=2);epp.join(timeout=2);voice.join(timeout=2);camera.join(timeout=2)


if __name__=="__main__":main()
