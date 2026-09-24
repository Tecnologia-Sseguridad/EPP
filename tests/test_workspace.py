"""Widget integration tests with fake services; no camera, voice or real data."""
from contextlib import ExitStack
import threading
import time
from types import SimpleNamespace
import tkinter as tk
import unittest
from unittest.mock import Mock, patch

import numpy as np

from engine import Latest
from sistema_final.ui.workspace import MainWindow
from sistema_final.ui.icons import render_icon


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.stack.enter_context(patch("sistema_final.ui.workspace.list_people", return_value=["Persona de prueba"]))
        self.stack.enter_context(patch("sistema_final.ui.workspace.get_required_epp", return_value=("casco", "chaleco", "guantes")))
        self.stack.enter_context(patch("sistema_final.ui.workspace.query_events", return_value=[]))
        self.stack.enter_context(patch("sistema_final.ui.main_window.record_event"))
        self.root = tk.Tk()
        self.root.withdraw()
        self.errors = []
        self.root.report_callback_exception = lambda *args: self.errors.append(args)
        self.stop = threading.Event()
        self.camera = SimpleNamespace(latest=Latest(), status="Prueba")
        self.facial = SimpleNamespace(latest=Latest(), status="Listo", enrolling=False,
                                      activate=Mock(), deactivate=Mock(), command=Mock())
        self.epp = SimpleNamespace(latest=Latest(), status="Listo", required=("casco", "chaleco", "guantes"),
                                   activate=Mock(), deactivate=Mock(), set_required=Mock())
        config = dict(camera="0", width=640, height=480, fps=30,
                      anti_spoofing_enabled=True, facial_threshold=.45, facial_margin=.08, epp_confidence=.28)
        self.app = MainWindow(self.root, self.camera, self.facial, self.epp, self.stop, config)
        self.root.update()

    def tearDown(self):
        self.stop.set()
        for task in self.root.tk.splitlist(self.root.tk.call("after", "info")):
            self.root.after_cancel(task)
        self.root.destroy()
        self.stack.close()

    def test_all_pages_construct_and_navigate(self):
        for page in range(3):
            self.app.tabs.select(page)
            self.root.update()
            if page == 2:
                for subpage in range(4):
                    self.app.admin_tabs.select(subpage)
                    self.root.update()
        self.assertFalse(self.errors)
        self.assertEqual(self.app.people_list.size(), 1)

    def test_stale_results_cannot_leave_complete_displayed(self):
        now = time.perf_counter()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.camera.latest.put(SimpleNamespace(sequence=1, captured=now, image=image))
        self.epp.latest.put(dict(sequence=1, captured=now - 5, preview=image,
                                people=[dict(id=1, overall="COMPLETO", decisions=dict(casco="si"))]))
        self.app.tabs.select(1)
        self.app.tick()
        self.assertEqual(self.app.epp_result.cget("text"), "Sin análisis reciente")
        self.assertFalse(self.errors)

    def test_new_presence_clears_voice_deduplication(self):
        self.app._presence_id = 1
        self.app.voice_epp_state["announced"] = ("COMPLETO", ())
        self.app._record_epp({"people": [{"id": 2, "overall": "VERIFICANDO", "decisions": {}}]})
        self.assertIsNone(self.app.voice_epp_state["announced"])
        self.assertEqual(self.app._presence_id, 2)

    def test_svg_icons_render_without_external_dependencies(self):
        for name in ("face", "shield", "users", "helmet", "vest", "glove", "settings"):
            icon = render_icon(name)
            self.assertEqual(icon.size, (20, 20))
            self.assertIsNotNone(icon.getbbox())


if __name__ == "__main__":
    unittest.main()
