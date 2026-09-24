"""Base única del sistema, migración y almacenamiento de evidencias."""
from datetime import datetime, timezone
from contextlib import contextmanager
import json
import hashlib
import shutil
import sqlite3
import uuid

import cv2

from .configuration import PROJECT_ROOT, ROOT

DATA_DIRECTORY = ROOT / "data"
DATABASE_PATH = DATA_DIRECTORY / "sistema.sqlite3"
LEGACY_DATABASE_PATH = PROJECT_ROOT / "data" / "people.sqlite3"
EVIDENCE_DIRECTORY = DATA_DIRECTORY / "evidence"
ENROLLMENT_DIRECTORY = DATA_DIRECTORY / "enrollments"
DEFAULT_REQUIREMENTS = ("casco", "chaleco", "guantes")


@contextmanager
def connect():
    database = sqlite3.connect(DATABASE_PATH, timeout=10)
    database.row_factory = sqlite3.Row
    database.execute("PRAGMA foreign_keys=ON")
    database.execute("PRAGMA journal_mode=WAL")
    try:
        yield database
        database.commit()
    except Exception:
        database.rollback()
        raise
    finally:
        database.close()


def initialize_database():
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    ENROLLMENT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    if not DATABASE_PATH.exists() and LEGACY_DATABASE_PATH.exists():
        # SQLite backup also includes committed pages that may still be in WAL.
        source = sqlite3.connect(LEGACY_DATABASE_PATH)
        target = sqlite3.connect(DATABASE_PATH)
        try:
            source.backup(target)
        finally:
            target.close()
            source.close()
    with connect() as database:
        database.executescript("""
            CREATE TABLE IF NOT EXISTS templates (name TEXT NOT NULL, model TEXT NOT NULL, vector BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS people (
                id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, event_type TEXT NOT NULL,
                person_name TEXT, status TEXT NOT NULL, helmet_status TEXT,
                vest_status TEXT, gloves_status TEXT, liveness_status TEXT,
                facial_score REAL, image_path TEXT, camera TEXT,
                details_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_events_person ON events(person_name);
            CREATE INDEX IF NOT EXISTS idx_events_type_status ON events(event_type, status);
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL
            );
        """)
        now = datetime.now(timezone.utc).isoformat()
        database.execute("INSERT OR IGNORE INTO schema_version VALUES(1, ?)", (now,))
        database.execute("""
            INSERT OR IGNORE INTO people(name, created_at, updated_at)
            SELECT DISTINCT name, ?, ? FROM templates
        """, (now, now))
        database.execute(
            "INSERT OR IGNORE INTO settings VALUES('required_epp', ?, ?)",
            (json.dumps(DEFAULT_REQUIREMENTS), now),
        )


def list_people():
    with connect() as database:
        return [row["name"] for row in database.execute(
            "SELECT name FROM people ORDER BY name COLLATE NOCASE"
        )]


def sync_people():
    now = datetime.now(timezone.utc).isoformat()
    with connect() as database:
        database.execute("""
            INSERT OR IGNORE INTO people(name, created_at, updated_at)
            SELECT DISTINCT name, ?, ? FROM templates
        """, (now, now))


def delete_person(name):
    with connect() as database:
        cursor = database.execute("DELETE FROM templates WHERE name=?", (name,))
        database.execute("DELETE FROM people WHERE name=?", (name,))
        deleted=cursor.rowcount
    folder=enrollment_folder(name)
    if folder.exists():shutil.rmtree(folder)
    return deleted


def enrollment_folder(name):
    return ENROLLMENT_DIRECTORY / hashlib.sha256(name.strip().casefold().encode("utf-8")).hexdigest()[:24]


def save_enrollment_images(name, images):
    folder=enrollment_folder(name);temporary=folder.with_name(folder.name+".tmp")
    if temporary.exists():shutil.rmtree(temporary)
    temporary.mkdir(parents=True,exist_ok=True)
    for index,image in enumerate(images[:8],1):cv2.imwrite(str(temporary/f"{index:02d}.jpg"),image,[cv2.IMWRITE_JPEG_QUALITY,90])
    if folder.exists():shutil.rmtree(folder)
    temporary.replace(folder)


def list_enrollment_images(name):
    folder=enrollment_folder(name)
    return sorted(folder.glob("*.jpg")) if folder.exists() else []


def get_required_epp():
    with connect() as database:
        row = database.execute("SELECT value FROM settings WHERE key='required_epp'").fetchone()
    values = json.loads(row["value"]) if row else list(DEFAULT_REQUIREMENTS)
    allowed = {"casco", "chaleco", "guantes", "antiparras", "botas"}
    return tuple(value for value in values if value in allowed)


def set_required_epp(values):
    ordered = [name for name in ("casco", "chaleco", "guantes", "antiparras", "botas") if name in values]
    if not ordered:
        raise ValueError("Debe existir al menos un EPP obligatorio")
    now = datetime.now(timezone.utc).isoformat()
    with connect() as database:
        database.execute("""
            INSERT INTO settings(key, value, updated_at) VALUES('required_epp', ?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
        """, (json.dumps(ordered, ensure_ascii=False), now))
    return tuple(ordered)


def record_event(event_type, status, frame=None, person_name=None, decisions=None,
                 liveness_status=None, facial_score=None, camera="0", details=None):
    timestamp = datetime.now(timezone.utc)
    image_path = None
    if frame is not None:
        folder = EVIDENCE_DIRECTORY / timestamp.strftime("%Y-%m-%d")
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{timestamp.strftime('%H%M%S_%f')}_{event_type}_{uuid.uuid4().hex[:8]}.jpg"
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if ok:
            temporary = target.with_suffix(".tmp")
            temporary.write_bytes(encoded.tobytes())
            temporary.replace(target)
            image_path = str(target.relative_to(ROOT))
    decisions = decisions or {}
    with connect() as database:
        cursor = database.execute("""
            INSERT INTO events(timestamp, event_type, person_name, status,
                helmet_status, vest_status, gloves_status, liveness_status,
                facial_score, image_path, camera, details_json)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (timestamp.isoformat(), event_type, person_name, status,
              decisions.get("casco"), decisions.get("chaleco"), decisions.get("guantes"),
              liveness_status, facial_score, image_path, str(camera),
              json.dumps(details or {}, ensure_ascii=False)))
        return cursor.lastrowid


def query_events(person="", event_type="Todos", status="Todos", date_from="", date_to="", limit=500, offset=0):
    clauses, parameters = [], []
    for value, clause in ((person.strip(), "person_name LIKE ?"),):
        if value:
            clauses.append(clause)
            parameters.append(f"%{value}%")
    if event_type != "Todos":
        clauses.append("event_type=?")
        parameters.append(event_type)
    if status != "Todos":
        clauses.append("status=?")
        parameters.append(status)
    if date_from.strip():
        clauses.append("date(timestamp, 'localtime') >= date(?)")
        parameters.append(date_from.strip())
    if date_to.strip():
        clauses.append("date(timestamp, 'localtime') <= date(?)")
        parameters.append(date_to.strip())
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    parameters.append(int(limit))
    parameters.append(max(0, int(offset)))
    with connect() as database:
        return database.execute(
            "SELECT * FROM events" + where + " ORDER BY timestamp DESC, id DESC LIMIT ? OFFSET ?", parameters
        ).fetchall()


def delete_events(event_ids=None):
    """Elimina eventos y sus evidencias. Sin IDs elimina el historial completo."""
    with connect() as database:
        if event_ids is None:
            rows = database.execute("SELECT image_path FROM events WHERE image_path IS NOT NULL").fetchall()
            deleted = database.execute("DELETE FROM events").rowcount
        else:
            ids = tuple(dict.fromkeys(int(value) for value in event_ids))
            if not ids:
                return 0
            placeholders = ",".join("?" for _ in ids)
            rows = database.execute(
                f"SELECT image_path FROM events WHERE id IN ({placeholders}) AND image_path IS NOT NULL", ids
            ).fetchall()
            deleted = database.execute(f"DELETE FROM events WHERE id IN ({placeholders})", ids).rowcount
    # Las rutas almacenadas siempre son relativas a ROOT. Se valida el destino
    # antes de borrar para que un dato corrupto no pueda afectar otros archivos.
    evidence_root = EVIDENCE_DIRECTORY.resolve()
    for row in rows:
        target = (ROOT / row["image_path"]).resolve()
        if target.is_relative_to(evidence_root) and target.is_file():
            target.unlink()
    return deleted
