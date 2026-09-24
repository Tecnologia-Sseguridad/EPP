import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
CONFIG_PATH = ROOT / "config.json"


def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def save_config(config):
    if not str(config.get("camera", "")).strip():
        raise ValueError("Indique una cámara.")
    if (config.get("width"), config.get("height")) not in ((640, 480), (1280, 720), (1920, 1080)):
        raise ValueError("Resolución no admitida.")
    if config.get("fps") not in (15, 25, 30, 60):
        raise ValueError("FPS no admitidos.")
    temporary = CONFIG_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(CONFIG_PATH)
