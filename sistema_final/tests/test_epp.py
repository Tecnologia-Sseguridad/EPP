import unittest
import time

from sistema_final.epp.service import (Person, add_evidence, assign_to_people,
                                       decide, framing_status, select_primary_person)


class EppLogicTests(unittest.TestCase):
    def test_positive_evidence_requires_temporal_confirmation(self):
        person = Person([0, 0, 100, 200], 0.9, 1)
        person.visible["casco"] = True
        person.evidence["casco"].extend([1, 1])
        self.assertEqual(decide(person, "casco"), "si")

    def test_invisible_zone_is_not_reported_as_missing(self):
        person = Person([0, 0, 100, 200], 0.9, 1)
        person.visible["guantes"] = False
        person.evidence["guantes"].extend([-1, -1])
        self.assertEqual(decide(person, "guantes"), "no visible")

    def test_detection_is_assigned_to_containing_person(self):
        people = {7: Person([0, 0, 100, 200], 0.9, 1)}
        detection = {"box": [20, 10, 80, 55], "confidence": 0.9, "name": "helmet"}
        self.assertIn("helmet", assign_to_people([detection], people)[7])

    def test_primary_person_ignores_smaller_false_detection(self):
        detections = [
            {"box": [40, 30, 500, 470], "confidence": 0.9, "name": "Person", "track_id": 1},
            {"box": [520, 40, 650, 380], "confidence": 0.8, "name": "Person", "track_id": 2},
        ]
        selected = select_primary_person(detections, (480, 680, 3))
        self.assertEqual(selected["track_id"], 1)

    def test_missing_detection_is_neutral(self):
        person = Person([120, 20, 500, 440], 0.9, 1)
        for _ in range(15):
            add_evidence(person, {}, (480, 640, 3))
        self.assertEqual(decide(person, "casco"), "verificando")
        self.assertEqual(decide(person, "chaleco"), "verificando")

    def test_explicit_negative_is_required_for_missing(self):
        person = Person([120, 20, 500, 440], 0.9, 1)
        detection = {"box": [180, 10, 330, 90], "confidence": 0.85, "name": "no_helmet"}
        for _ in range(2):
            add_evidence(person, {"no_helmet": [detection]}, (480, 640, 3))
        self.assertEqual(decide(person, "casco"), "no")

    def test_one_glove_does_not_confirm_both_hands(self):
        person = Person([120, 20, 500, 440], 0.9, 1)
        glove = {"box": [150, 230, 200, 290], "confidence": 0.9, "name": "gloves"}
        for _ in range(3):
            add_evidence(person, {"gloves": [glove]}, (480, 640, 3))
        self.assertEqual(decide(person, "guantes"), "verificando")

    def test_two_gloves_confirm_both_hands(self):
        person = Person([120, 20, 500, 440], 0.9, 1)
        gloves = [
            {"box": [140, 230, 190, 290], "confidence": 0.9, "name": "gloves"},
            {"box": [430, 230, 480, 290], "confidence": 0.88, "name": "gloves"},
        ]
        for _ in range(2):
            add_evidence(person, {"gloves": gloves}, (480, 640, 3))
        self.assertEqual(decide(person, "guantes"), "si")

    def test_framing_requires_useful_distance(self):
        shape = (480, 640, 3)
        self.assertEqual(framing_status([260, 130, 380, 330], shape), "acercate")
        self.assertEqual(framing_status([20, 0, 620, 479], shape), "alejate")
        self.assertEqual(framing_status([180, 20, 460, 440], shape), "correcto")

    def test_confirmed_state_survives_brief_loss_but_expires(self):
        person = Person([120, 20, 500, 440], 0.9, 1)
        helmet = {"box": [180, 10, 330, 90], "confidence": 0.9, "name": "helmet"}
        for _ in range(2):
            add_evidence(person, {"helmet": [helmet]}, (480, 640, 3))
        self.assertEqual(decide(person, "casco"), "si")
        add_evidence(person, {}, (480, 640, 3))
        self.assertEqual(decide(person, "casco"), "si")
        person.last_explicit["casco"] = time.perf_counter() - 3.0
        self.assertEqual(decide(person, "casco"), "verificando")

if __name__ == "__main__":
    unittest.main()
