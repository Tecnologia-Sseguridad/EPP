import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from sistema_final.epp.service import EppService


class Tensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def int(self):
        return self

    def tolist(self):
        return self.values


class PipelineTests(unittest.TestCase):
    def test_raw_predictions_without_track_ids_confirm_equipment(self):
        clock = [10.]
        remaining = [5]
        stop = Mock()
        stop.is_set.side_effect = lambda: remaining[0] <= 0
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        def next_frame():
            remaining[0] -= 1
            clock[0] += .2
            return SimpleNamespace(sequence=5 - remaining[0], captured=clock[0], image=image)

        camera = SimpleNamespace(latest=SimpleNamespace(get=next_frame))
        boxes = SimpleNamespace(
            xyxy=Tensor([[120, 20, 500, 440], [200, 20, 330, 90], [180, 140, 430, 310],
                         [130, 230, 190, 290], [430, 230, 490, 290]]),
            conf=Tensor([.95] * 5), cls=Tensor([0, 1, 2, 3, 3]))
        model = Mock()
        model.names = {0: "Person", 1: "helmet", 2: "vest", 3: "gloves"}
        model.predict.return_value = [SimpleNamespace(boxes=boxes)]
        service = EppService(camera, stop)
        service.activate()
        with patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=lambda path: model)}):
            with patch("sistema_final.epp.service.time.perf_counter", side_effect=lambda: clock[0]):
                service.run()
        result = service.latest.get()
        self.assertEqual(result["people"][0]["overall"], "COMPLETO")
        self.assertEqual(model.predict.call_count, 5)
        model.track.assert_not_called()

    def test_changed_requirements_clear_published_result(self):
        service = EppService(None, threading.Event())
        service.latest.put({"people": [{"overall": "COMPLETO"}]})
        service.set_required(("casco", "guantes"))
        self.assertIsNone(service.latest.get())
        self.assertTrue(service.reset_requested.is_set())


if __name__ == "__main__":
    unittest.main()
