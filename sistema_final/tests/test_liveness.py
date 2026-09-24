import unittest

from sistema_final.facial.anti_spoofing import LivenessDecision


class LivenessDecisionTests(unittest.TestCase):
    def test_does_not_accept_a_single_frame(self):
        decision = LivenessDecision()
        state, _ = decision.update(0.99)
        self.assertEqual(state, "verificando")

    def test_accepts_only_consistent_real_evidence(self):
        decision = LivenessDecision()
        states = [decision.update(score)[0] for score in (0.91, 0.88, 0.94, 0.90)]
        self.assertEqual(states[-1], "real")

    def test_rejects_consistent_strong_spoof_evidence(self):
        decision = LivenessDecision()
        states = [decision.update(score)[0] for score in (0.05, 0.08, 0.04, 0.06)]
        self.assertEqual(states[-1], "falso")

    def test_ambiguous_evidence_remains_inconclusive(self):
        decision = LivenessDecision()
        states = [decision.update(score)[0] for score in (0.40, 0.55, 0.42, 0.52)]
        self.assertEqual(states[-1], "inconcluso")


if __name__ == "__main__":
    unittest.main()
