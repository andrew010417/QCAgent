import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from config import settings

# Paste your NCBI API key here if you want to use NCBI literature APIs directly from db.py.
# If you set NCBI_API_KEY in api_key.py or the environment, that value will override this placeholder.
NCBI_API_KEY: str = getattr(settings, "NCBI_API_KEY", "") or "<paste your NCBI API key here>"

DB_PATH = Path(settings.STORAGE_PATH) / "workflow_results.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                input_text TEXT NOT NULL,
                category TEXT NOT NULL,
                qc_output TEXT,
                report_output TEXT
            )
            """
        )
        conn.commit()


def save_workflow_result(input_text: str, category: str, qc_output: str, report_output: str | None) -> int:
    created_at = datetime.utcnow().isoformat() + "Z"
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO workflow_runs (created_at, input_text, category, qc_output, report_output) VALUES (?, ?, ?, ?, ?)",
            (created_at, input_text, category, qc_output, report_output),
        )
        conn.commit()
        return cursor.lastrowid


def fetch_workflow_run(run_id: int) -> dict[str, Any] | None:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, created_at, input_text, category, qc_output, report_output FROM workflow_runs WHERE id = ?", (run_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "created_at": row[1],
            "input_text": row[2],
            "category": row[3],
            "qc_output": row[4],
            "report_output": row[5],
        }
