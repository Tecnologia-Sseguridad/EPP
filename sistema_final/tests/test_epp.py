import unittest
import numpy as np

from sistema_final.epp.temporal import Evidence, overall_status
from sistema_final.epp.service import (
    Person, Presence, add_evidence, decide, assign_to_people,
    framing_status, select_primary_person, remove_duplicates, draw_person,
)


def detection(name, box, confidence=.9):
    return {"name": name, "box": box, "confidence": confidence, "track_id": None}


class EppLogicTests(unittest.TestCase):
    def setUp(self):
        self.person = Person([120, 20, 500, 440], .9, 1)
        self.helmet = detection("helmet", [180, 10, 330, 90])
        self.vest = detection("vest", [180, 130, 400, 310])
        self.left = detection("gloves", [140, 230, 190, 290])
        self.right = detection("gloves", [430, 230, 480, 290])
        self.shape = (480, 640, 3)

    def observe(self, detections, start=10, count=4):
        for i in range(count):
            add_evidence(self.person, detections, self.shape, now=start + i * .2)
        return start + (count - 1) * .2

    def test_confirmation_requires_multiple_frames_and_time(self):
        evidence = Evidence()
        evidence.observe(1, 10)
        evidence.observe(1, 10.01)
        evidence.observe(1, 10.02)
        self.assertEqual(evidence.decision(10.02), "verificando")
        evidence.observe(1, 10.4)
        self.assertEqual(evidence.decision(10.4), "si")

    def test_rendering_cannot_reconfirm_expired_evidence(self):
        now = self.observe({"helmet": [self.helmet]})
        self.assertEqual(decide(self.person, "casco", now), "si")
        for _ in range(30):
            self.assertEqual(decide(self.person, "casco", now + 3), "verificando")

    def test_brief_loss_does_not_flicker(self):
        now = self.observe({"helmet": [self.helmet]})
        add_evidence(self.person, {}, self.shape, now=now + .2)
        self.assertEqual(decide(self.person, "casco", now + .2), "si")

    def test_no_evidence_never_means_missing(self):
        now = self.observe({}, count=20)
        self.assertEqual(decide(self.person, "casco", now), "verificando")

    def test_explicit_negative_is_required(self):
        now = self.observe({"no_helmet": [detection("no_helmet", [180, 10, 330, 90])]})
        self.assertEqual(decide(self.person, "casco", now), "no")

    def test_weak_negative_is_ignored(self):
        now = self.observe({"none": [detection("none", [180, 130, 400, 310], .32)]}, count=10)
        self.assertEqual(decide(self.person, "chaleco", now), "verificando")

    def test_conflicting_helmet_classes_are_inconclusive(self):
        now = self.observe({"helmet": [self.helmet],
                            "no_helmet": [detection("no_helmet", [180, 10, 330, 90], .85)]})
        self.assertEqual(decide(self.person, "casco", now), "verificando")

    def test_one_glove_is_not_a_pair(self):
        now = self.observe({"gloves": [self.left]})
        self.assertEqual(decide(self.person, "guantes", now), "verificando")

    def test_one_glove_moving_across_body_is_not_a_pair(self):
        for i in range(12):
            add_evidence(self.person, {"gloves": [self.left if i % 2 else self.right]}, self.shape, now=10 + .2 * i)
        self.assertEqual(decide(self.person, "guantes", 12.2), "verificando")

    def test_two_distinct_gloves_are_confirmed(self):
        now = self.observe({"gloves": [self.left, self.right]})
        self.assertEqual(decide(self.person, "guantes", now), "si")

    def test_duplicate_glove_boxes_are_not_a_pair(self):
        now = self.observe({"gloves": [self.left, self.left.copy()]})
        self.assertEqual(decide(self.person, "guantes", now), "verificando")
        self.assertEqual(len(remove_duplicates([self.left, self.left.copy()])), 1)

    def test_bare_hand_prevents_compliance(self):
        now = self.observe({"gloves": [self.left, self.right],
                            "no_gloves": [detection("no_gloves", [420, 230, 480, 290])]})
        self.assertEqual(decide(self.person, "guantes", now), "no")

    def test_removal_changes_only_the_affected_equipment(self):
        self.observe({"helmet": [self.helmet], "vest": [self.vest]})
        now = self.observe({"vest": [self.vest], "no_helmet": [
            detection("no_helmet", [180, 10, 330, 90])]}, start=10.8, count=12)
        self.assertEqual(decide(self.person, "casco", now), "no")
        self.assertEqual(decide(self.person, "chaleco", now), "si")

    def test_untracked_person_is_accepted(self):
        selected = select_primary_person([detection("Person", [40, 30, 500, 470])], self.shape)
        self.assertIsNotNone(selected)

    def test_helmet_above_box_is_assigned(self):
        people = {1: Person([100, 100, 300, 450], .9, 1)}
        helmet = detection("helmet", [145, 42, 255, 112])
        self.assertIn("helmet", assign_to_people([helmet], people)[1])

    def test_glove_outside_box_is_assigned(self):
        people = {1: Person([100, 40, 300, 440], .9, 1)}
        self.assertIn("gloves", assign_to_people([
            detection("gloves", [55, 245, 115, 315])], people)[1])

    def test_helmet_held_at_waist_is_not_assigned(self):
        self.assertNotIn("helmet", assign_to_people([
            detection("helmet", [180, 340, 280, 420])], {1: self.person})[1])

    def test_overall_requires_all_requested_items(self):
        self.assertEqual(overall_status({"casco": "si", "guantes": "verificando"}), "VERIFICANDO")
        self.assertEqual(overall_status({}), "VERIFICANDO")
        self.assertEqual(overall_status({"casco": "si"}, present=False), "VERIFICANDO")
        self.assertEqual(overall_status({"casco": "si", "guantes": "si"}), "COMPLETO")

    def test_small_movement_keeps_presence(self):
        session = Presence()
        first = session.update(detection("Person", [100, 30, 400, 470]), 10)
        second = session.update(detection("Person", [110, 35, 410, 475]), 10.2)
        self.assertIs(first, second)

    def test_missing_person_never_emits_old_compliance(self):
        session = Presence()
        session.update(detection("Person", [100, 30, 400, 470]), 10)
        self.assertIsNone(session.update(None, 10.1))

    def test_long_absence_starts_new_evidence(self):
        session = Presence()
        first = session.update(detection("Person", [100, 30, 400, 470]), 10)
        second = session.update(detection("Person", [100, 30, 400, 470]), 11)
        self.assertNotEqual(first.display_id, second.display_id)
        self.assertEqual(decide(second, "casco", 11), "no visible")

    def test_large_displacement_starts_new_presence(self):
        session = Presence()
        first = session.update(detection("Person", [0, 20, 180, 450]), 10)
        second = session.update(detection("Person", [430, 20, 620, 450]), 10.2)
        self.assertNotEqual(first.display_id, second.display_id)

    def test_legs_at_border_do_not_block_vest_evidence(self):
        self.person.raw_box = [120, 20, 500, 480]
        now = self.observe({"vest": [self.vest]})
        self.assertEqual(decide(self.person, "chaleco", now), "si")

    def test_outline_is_monochrome(self):
        frame = np.zeros(self.shape, dtype=np.uint8)
        draw_person(frame, self.person)
        self.assertTrue(np.array_equal(frame[:, :, 0], frame[:, :, 1]))
        self.assertTrue(np.array_equal(frame[:, :, 1], frame[:, :, 2]))


if __name__ == "__main__":
    unittest.main()
