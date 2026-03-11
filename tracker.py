"""
Experiment tracker backed by SQLite.
Replaces TSV-based logging with structured storage in experiments.db.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments.db")


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            commit          TEXT    NOT NULL,
            val_bpb         REAL    NOT NULL,
            memory_gb       REAL    NOT NULL,
            status          TEXT    NOT NULL,
            description     TEXT    NOT NULL,
            config_json     TEXT,
            checkpoint_path TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_experiment(commit, val_bpb, memory_gb, status, description,
                   config_dict=None, checkpoint_path=None):
    conn = _connect()
    timestamp = datetime.now(timezone.utc).isoformat()
    config_json = json.dumps(config_dict) if config_dict is not None else None
    cursor = conn.execute(
        """
        INSERT INTO experiments
            (timestamp, commit, val_bpb, memory_gb, status, description,
             config_json, checkpoint_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (timestamp, commit[:7], val_bpb, memory_gb, status, description,
         config_json, checkpoint_path),
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def get_all_experiments():
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM experiments ORDER BY timestamp"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_best_experiment():
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM experiments WHERE status = 'keep' "
        "ORDER BY val_bpb ASC LIMIT 1"
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_experiment_by_commit(commit):
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM experiments WHERE commit = ? ORDER BY timestamp",
        (commit[:7],),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_checkpoint(model, config, commit, directory="checkpoints"):
    import torch
    dirpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), directory)
    os.makedirs(dirpath, exist_ok=True)
    filepath = os.path.join(dirpath, f"{commit[:7]}.pt")
    torch.save({"config": config, "model_state_dict": model.state_dict()}, filepath)
    return filepath
