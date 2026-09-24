import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sistema_final.core import database


class SystemDatabaseTests(unittest.TestCase):
    def test_migration_event_image_filters_and_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            legacy = root / "legacy.sqlite3"
            connection = sqlite3.connect(legacy)
            try:
                connection.execute("CREATE TABLE templates(name TEXT, model TEXT, vector BLOB)")
                connection.execute("INSERT INTO templates VALUES('Ana', 'sface', X'00')")
                connection.commit()
            finally:
                connection.close()
            data = root / "data"
            replacements = {
                "ROOT": root,
                "DATA_DIRECTORY": data,
                "DATABASE_PATH": data / "sistema.sqlite3",
                "LEGACY_DATABASE_PATH": legacy,
                "EVIDENCE_DIRECTORY": data / "evidence",
            }
            with patch.multiple(database, **replacements):
                database.initialize_database()
                self.assertEqual(database.list_people(), ["Ana"])
                database.set_required_epp(("casco", "guantes"))
                self.assertEqual(database.get_required_epp(), ("casco", "guantes"))
                event_id = database.record_event(
                    "epp", "INCOMPLETO", frame=np.zeros((20, 20, 3), dtype=np.uint8),
                    person_name="Ana", decisions={"casco": "si", "guantes": "no"},
                )
                rows = database.query_events(person="Ana", event_type="epp", status="INCOMPLETO")
                self.assertEqual([row["id"] for row in rows], [event_id])
                evidence = root / rows[0]["image_path"]
                self.assertTrue(evidence.is_file())
                self.assertEqual(database.delete_events([event_id]), 1)
                self.assertFalse(evidence.exists())
                self.assertEqual(database.query_events(), [])


if __name__ == "__main__":
    unittest.main()
