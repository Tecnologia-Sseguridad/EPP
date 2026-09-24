import unittest

from sistema_final.voices import epp_message


class VoiceMessageTests(unittest.TestCase):
    def test_complete_message(self):
        text = epp_message("COMPLETO", {"casco": "si", "chaleco": "si"}, ("casco", "chaleco"))
        self.assertIn("completo", text)
        self.assertIn("Puede continuar", text)

    def test_partial_message_lists_missing_items(self):
        text = epp_message("INCOMPLETO", {"casco": "si", "chaleco": "no", "guantes": "no"},
                           ("casco", "chaleco", "guantes"))
        self.assertIn("chaleco reflectante y guantes", text)

    def test_no_epp_message(self):
        text = epp_message("INCOMPLETO", {"casco": "no", "chaleco": "no"}, ("casco", "chaleco"))
        self.assertIn("sin equipo de protección personal", text)


if __name__ == "__main__":
    unittest.main()
