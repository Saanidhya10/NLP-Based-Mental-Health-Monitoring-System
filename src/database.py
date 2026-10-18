"""SQLite mood-history store (the synopsis lists MongoDB/MySQL/Firebase as options; SQLite needs no server).

Privacy by design (paper, Sec. 10.2):
  * nothing is stored without explicit, revocable consent
  * by default only derived scores + timestamps are kept, never raw text
  * users can view, export and delete their own data
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.trends import PROB_COLS, build_prediction_frame, negative_scores
from src.utils import ID2LABEL, LABELS, load_config, resolve_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (user_id TEXT PRIMARY KEY, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS consent (user_id TEXT PRIMARY KEY, granted INTEGER NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    emotion TEXT NOT NULL,
    confidence REAL NOT NULL,
    negative_score REAL NOT NULL,
    probs TEXT NOT NULL,
    text TEXT
);
CREATE INDEX IF NOT EXISTS idx_pred_user_time ON predictions(user_id, timestamp);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MoodDatabase:
    def __init__(self, path: str | Path | None = None, cfg: dict | None = None):
        if path is None:
            cfg = cfg or load_config()
            path = resolve_path(cfg["database"]["path"])
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._memory_conn = sqlite3.connect(":memory:") if str(path) == ":memory:" else None
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = self._memory_conn or sqlite3.connect(self.path)
        try:
            yield conn
            conn.commit()
        finally:
            if not self._memory_conn:
                conn.close()

    # ---- consent -----------------------------------------------------------------
    def set_consent(self, user_id: str, granted: bool, delete_data_on_revoke: bool = False) -> None:
        with self._conn() as c:
            c.execute("INSERT OR IGNORE INTO users VALUES (?, ?)", (user_id, _now()))
            c.execute("INSERT INTO consent VALUES (?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET "
                      "granted=excluded.granted, updated_at=excluded.updated_at", (user_id, int(granted), _now()))
        if not granted and delete_data_on_revoke:
            self.delete_user_data(user_id, keep_user=True)

    def has_consent(self, user_id: str) -> bool:
        with self._conn() as c:
            row = c.execute("SELECT granted FROM consent WHERE user_id=?", (user_id,)).fetchone()
        return bool(row and row[0])

    # ---- write -------------------------------------------------------------------
    def add_prediction(self, user_id: str, probs, timestamp=None, text: str | None = None,
                       store_text: bool = False) -> None:
        """Store one prediction. Raises PermissionError without consent. Raw text only if store_text=True."""
        if not self.has_consent(user_id):
            raise PermissionError(f"User '{user_id}' has not granted consent; nothing was stored.")
        probs = np.asarray(probs, dtype=float).ravel()
        ts = pd.Timestamp(timestamp or datetime.now()).isoformat()
        with self._conn() as c:
            c.execute(
                "INSERT INTO predictions (user_id, timestamp, emotion, confidence, negative_score, probs, text) "
                "VALUES (?,?,?,?,?,?,?)",
                (user_id, ts, ID2LABEL[int(probs.argmax())], float(probs.max()),
                 float(negative_scores(probs[None, :])[0]), json.dumps(probs.tolist()),
                 text if (store_text and text) else None))

    # ---- read --------------------------------------------------------------------
    def get_predictions(self, user_id: str, since=None) -> pd.DataFrame:
        """Message-level frame in the format expected by trends.aggregate / mood_summary."""
        q, args = "SELECT timestamp, probs FROM predictions WHERE user_id=?", [user_id]
        if since is not None:
            q += " AND timestamp >= ?"
            args.append(pd.Timestamp(since).isoformat())
        with self._conn() as c:
            rows = c.execute(q + " ORDER BY timestamp", args).fetchall()
        if not rows:
            return pd.DataFrame(columns=["timestamp", *PROB_COLS, "neg_score", "emotion", "confidence"])
        return build_prediction_frame([r[0] for r in rows], [json.loads(r[1]) for r in rows], user_id=user_id)

    def export_user_data(self, user_id: str) -> dict:
        with self._conn() as c:
            cons = c.execute("SELECT granted, updated_at FROM consent WHERE user_id=?", (user_id,)).fetchone()
            preds = c.execute("SELECT timestamp, emotion, confidence, negative_score, probs, text "
                              "FROM predictions WHERE user_id=? ORDER BY timestamp", (user_id,)).fetchall()
        return {
            "user_id": user_id,
            "consent": {"granted": bool(cons[0]), "updated_at": cons[1]} if cons else None,
            "predictions": [
                {"timestamp": t, "emotion": e, "confidence": cf, "negative_score": ns,
                 "probabilities": dict(zip(LABELS, json.loads(p))), "text": tx}
                for t, e, cf, ns, p, tx in preds],
        }

    def delete_user_data(self, user_id: str, keep_user: bool = False) -> int:
        with self._conn() as c:
            n = c.execute("DELETE FROM predictions WHERE user_id=?", (user_id,)).rowcount
            if not keep_user:
                c.execute("DELETE FROM consent WHERE user_id=?", (user_id,))
                c.execute("DELETE FROM users WHERE user_id=?", (user_id,))
        return n

    def list_users(self) -> list[str]:
        with self._conn() as c:
            return [r[0] for r in c.execute("SELECT user_id FROM users ORDER BY user_id")]
