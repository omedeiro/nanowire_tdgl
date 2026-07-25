"""SQLite-backed persistence for projects and jobs.

Deliberately simple (no Redis/Celery) so the server is friendly to
self-hosting behind a Cloudflare Tunnel.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, List, Optional
from uuid import uuid4

_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    document TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    simulation_id TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    artifact_path TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
"""


class Store:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # -- projects ------------------------------------------------------------
    def put_project(self, project_id: str, document: dict) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO projects(id, document, updated_at) VALUES(?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET document=excluded.document, "
                "updated_at=excluded.updated_at",
                (project_id, json.dumps(document), time.time()),
            )

    def get_project(self, project_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT document FROM projects WHERE id=?", (project_id,)
            ).fetchone()
        return json.loads(row["document"]) if row else None

    def list_projects(self) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT document FROM projects ORDER BY updated_at DESC")
            return [json.loads(r["document"]) for r in rows]

    def delete_project(self, project_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM projects WHERE id=?", (project_id,))
            return cur.rowcount > 0

    # -- jobs ------------------------------------------------------------------
    def create_job(self, project_id: str, simulation_id: str) -> str:
        job_id = uuid4().hex
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs(id, project_id, simulation_id, status, created_at, "
                "updated_at) VALUES(?,?,?,?,?,?)",
                (job_id, project_id, simulation_id, "pending", now, now),
            )
        return job_id

    def update_job(
        self,
        job_id: str,
        status: str,
        *,
        error: Optional[str] = None,
        artifact_path: Optional[str] = None,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status=?, error=?, artifact_path=COALESCE(?, "
                "artifact_path), updated_at=? WHERE id=?",
                (status, error, artifact_path, time.time(), job_id),
            )

    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None
