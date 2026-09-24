import unittest
from validar_epp import summarize


class EvaluationTests(unittest.TestCase):
    def test_abstentions_are_not_reported_as_correct(self):
        result = summarize([
            dict(item="casco", expected="si", predicted="si"),
            dict(item="casco", expected="no", predicted="si"),
            dict(item="casco", expected="si", predicted="verificando"),
            dict(item="casco", expected="no", predicted="no"),
        ])["casco"]
        self.assertEqual(result["falsos_cumplimientos"], 1)
        self.assertEqual(result["sin_conclusion"], 1)
        self.assertEqual(result["precision_del_cumplimiento"], .5)
        self.assertEqual(result["acierto_incluyendo_abstenciones"], .5)

    def test_no_positive_predictions_has_no_precision_estimate(self):
        result = summarize([dict(item="guantes", expected="si", predicted="verificando")])
        self.assertIsNone(result["guantes"]["precision_del_cumplimiento"])


if __name__ == "__main__":
    unittest.main()
