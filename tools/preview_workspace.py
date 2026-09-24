"""Render the real desktop views with synthetic services, never a live camera."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import runpy
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
from PIL import ImageGrab

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    output = Path(args.output) if args.output else Path(tempfile.mkdtemp(prefix="epp_ui_preview_"))
    output.mkdir(parents=True, exist_ok=True)
    tests = runpy.run_path(str(ROOT / "tests" / "test_workspace.py"))
    fixture = tests["WorkspaceTests"]("test_all_pages_construct_and_navigate")
    fixture.setUp()
    try:
        root, app = fixture.root, fixture.app
        root.geometry("1280x800+10+10")
        root.deiconify()
        root.lift()
        root.update()
        # A neutral synthetic camera image, with no user data.
        image = np.full((480, 640, 3), 55, dtype=np.uint8)
        stamp = time.perf_counter()
        fixture.camera.latest.put(SimpleNamespace(sequence=1, captured=stamp, image=image))
        fixture.epp.latest.put(dict(sequence=1, captured=stamp, preview=image,
            annotation_mask=np.zeros((480, 640), dtype=bool), inference_ms=120,
            guidance="Deje visibles la cabeza, el torso y ambas manos",
            people=[dict(id=1, overall="VERIFICANDO",
                         decisions={"casco": "si", "chaleco": "si", "guantes": "verificando"})]))
        for name, tab, sub in (("epp", 1, 0), ("facial", 0, 0),
                                ("personas", 2, 0), ("registros", 2, 1), ("configuracion", 2, 2)):
            app.tabs.select(tab)
            if tab == 2:
                app.admin_tabs.select(sub)
            root.update()
            app.last_camera_sequence = -1
            app.tick()
            root.update()
            bbox = (root.winfo_rootx(), root.winfo_rooty(),
                    root.winfo_rootx() + root.winfo_width(), root.winfo_rooty() + root.winfo_height())
            ImageGrab.grab(bbox=bbox).save(output / (name + ".png"))
        if fixture.errors:
            raise RuntimeError(fixture.errors)
        print(output)
    finally:
        fixture.tearDown()


if __name__ == "__main__":
    main()
