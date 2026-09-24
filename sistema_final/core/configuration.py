import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
CONFIG_PATH = ROOT / "config.json"


def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
