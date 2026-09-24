import tempfile
from pathlib import Path
import unittest

import numpy as np

from engine import Latest, Store, StableIdentity, match, normalize, classify_trial


class EngineTests(unittest.TestCase):
    def test_latest_discards_backlog(self):
        latest = Latest()
        for i in range(1000):
            latest.put(i)
        self.assertEqual(latest.get(), 999)

    def test_runner_up_is_different_person(self):
        query = normalize([1, 0])
        gallery = {"Ana": np.array([[1, 0], [.99, .01]], dtype=np.float32), "Luis": np.array([[0, 1]], dtype=np.float32)}
        self.assertEqual(match(query, gallery)[0], "Ana")
        gallery["Luis"] = np.array([[.98, .02]], dtype=np.float32)
        self.assertIsNone(match(query, gallery)[0])

    def test_unknown_and_empty_gallery(self):
        self.assertIsNone(match(normalize([0, 1]), {"Ana": np.array([[1, 0]], dtype=np.float32)})[0])
        self.assertIsNone(match(normalize([0, 1]), {})[0])

    def test_identity_requires_temporal_continuity(self):
        stable = StableIdentity()
        vector = normalize([1, 0])
        self.assertIsNone(stable.update("Ana", vector, 1))
        self.assertIsNone(stable.update("Ana", vector, 1.1))
        self.assertEqual(stable.update("Ana", vector, 1.2), "Ana")
        self.assertIsNone(stable.update("Ana", vector, 2))
        stable.update(None, vector, 2.1)
        self.assertIsNone(stable.update("Ana", vector, 2.2))

    def test_storage_replacement_persists(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "people.db"
            store = Store(path)
            vector = normalize(np.ones(128))
            store.save("Ana", [vector] * 8)
            store.save("Ana", [vector] * 3)
            store.close()
            store = Store(path)
            self.assertEqual(store.gallery()["Ana"].shape, (3, 128))
            store.close()

    def test_invalid_embeddings_rejected(self):
        for vector in ([0, 0], [float("nan"), 1]):
            with self.assertRaises(ValueError):
                normalize(vector)

    def test_evaluation_does_not_hide_wrong_identity(self):
        self.assertEqual(classify_trial("Ana", {"Ana", "Luis"}, 10), "identidad_incorrecta")
        self.assertEqual(classify_trial("Ana", {"Ana"}, 10), "identificacion_correcta")
        self.assertEqual(classify_trial("Ana", set(), 10), "falso_rechazo")

    def test_unknown_evaluation_requires_valid_observations(self):
        self.assertEqual(classify_trial("(Desconocido)", set(), 0), "sin_muestra_valida")
        self.assertEqual(classify_trial("(Desconocido)", set(), 10), "rechazo_correcto")
        self.assertEqual(classify_trial("(Desconocido)", {"Ana"}, 10), "falsa_aceptacion")


if __name__ == "__main__":
    unittest.main()
