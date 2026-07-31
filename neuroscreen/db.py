"""
SQLite storage for real-person screenings.

Every subject screened via ``neuroscreen.detect`` is persisted here: who they
are (patient_id/name/age/sex, as supplied), what was measured, and what the
model concluded. This gives a clinician a longitudinal history per patient
instead of a one-off printout.

The database file itself (default: ``data/neuroscreen.db``) is local state,
not code -- it is git-ignored. It may contain real names and health data, so
it must never be committed or pushed to a public repository.
"""

from __future__ import annotations

import datetime
import json
import os
import sqlite3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(ROOT, "data", "neuroscreen.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id  TEXT PRIMARY KEY,
    name        TEXT,
    age         REAL,
    sex         TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS screenings (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id        TEXT NOT NULL REFERENCES patients(patient_id),
    timestamp         TEXT NOT NULL,
    model_source      TEXT,
    prediction        TEXT,
    confidence        REAL,
    risk_level        TEXT,
    recommendation    TEXT,
    probabilities     TEXT,
    top_factors       TEXT,
    features_used     TEXT,
    features_imputed  TEXT
);

CREATE INDEX IF NOT EXISTS idx_screenings_patient ON screenings(patient_id, timestamp);
"""


def connect(path: str = DEFAULT_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def upsert_patient(conn, patient_id: str, name: str | None = None,
                    age: float | None = None, sex: str | None = None) -> None:
    row = conn.execute("SELECT patient_id FROM patients WHERE patient_id=?",
                        (patient_id,)).fetchone()
    now = _now()
    if row is None:
        conn.execute(
            "INSERT INTO patients (patient_id, name, age, sex, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?)", (patient_id, name, age, sex, now, now))
    else:
        # Refresh any newly-supplied identity fields without blanking existing ones.
        conn.execute(
            "UPDATE patients SET "
            "name=COALESCE(?, name), age=COALESCE(?, age), sex=COALESCE(?, sex), "
            "updated_at=? WHERE patient_id=?",
            (name, age, sex, now, patient_id))
    conn.commit()


def save_screening(conn, patient_id: str, model_source: str, result: dict,
                    features_used: list, features_imputed: list) -> int:
    cur = conn.execute(
        "INSERT INTO screenings (patient_id, timestamp, model_source, prediction, "
        "confidence, risk_level, recommendation, probabilities, top_factors, "
        "features_used, features_imputed) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (patient_id, _now(), model_source, result["prediction"], result["confidence"],
         result["risk_level"], result["recommendation"],
         json.dumps(result["probabilities"]), json.dumps(result["top_factors"]),
         json.dumps(features_used), json.dumps(features_imputed)))
    conn.commit()
    return cur.lastrowid


def patient_history(conn, patient_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM screenings WHERE patient_id=? ORDER BY timestamp DESC",
        (patient_id,)).fetchall()
    return [dict(r) for r in rows]


def list_patients(conn) -> list[dict]:
    rows = conn.execute("""
        SELECT p.patient_id, p.name, p.age, p.sex,
               COUNT(s.id) AS n_screenings,
               MAX(s.timestamp) AS last_screened,
               (SELECT prediction FROM screenings
                WHERE patient_id = p.patient_id
                ORDER BY timestamp DESC LIMIT 1) AS last_prediction
        FROM patients p LEFT JOIN screenings s ON s.patient_id = p.patient_id
        GROUP BY p.patient_id ORDER BY last_screened DESC
    """).fetchall()
    return [dict(r) for r in rows]
