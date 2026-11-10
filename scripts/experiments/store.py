"""SQLite transactions are the authority; JSON files are portable result copies."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid

from . import TERMINAL
from .schema import canonical, digest, identifier


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(canonical(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.path = self.root / ".scratch/experiments"
        self.path.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path.chmod(0o700)
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS jobs (seq INTEGER PRIMARY KEY AUTOINCREMENT, "
                       "id TEXT UNIQUE NOT NULL, fingerprint TEXT NOT NULL, record TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path / "queue.sqlite3", timeout=30)
        try:
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    def directory(self, job_id):
        return self.path / "jobs" / identifier(job_id)

    def submit(self, payload):
        job_id = payload["request"]["id"]
        fingerprint = digest(payload)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT fingerprint, record FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row:
                if row[0] != fingerprint:
                    raise ValueError("id already belongs to a different frozen request/source; use retry or a new id")
                return json.loads(row[1])
            record = dict(id=job_id, state="queued", submitted_at=time.time(), payload=payload,
                          cancel_requested=False, result=None, unit=None)
            db.execute("INSERT INTO jobs(id,fingerprint,record) VALUES(?,?,?)",
                       (job_id, fingerprint, canonical(record).decode()))
            return record

    def jobs(self):
        with self.connect() as db:
            return [dict(json.loads(row[1]), sequence=row[0])
                    for row in db.execute("SELECT seq,record FROM jobs ORDER BY seq")]

    def get(self, job_id):
        identifier(job_id)
        with self.connect() as db:
            row = db.execute("SELECT record FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise ValueError("unknown experiment id: " + job_id)
        return json.loads(row[0])

    def update(self, job_id, function):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT record FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise ValueError("unknown experiment id: " + job_id)
            record = json.loads(row[0])
            function(record)
            db.execute("UPDATE jobs SET record=? WHERE id=?", (canonical(record).decode(), job_id))
            return record

    def claim(self, job_id, unit):
        def change(record):
            if record["state"] != "queued" or record["cancel_requested"]:
                raise ValueError("job no longer queued")
            record.update(state="running", started_at=time.time(), unit=unit)
        return self.update(job_id, change)

    def cancel(self, job_id):
        def change(record):
            if record["state"] not in TERMINAL:
                record["cancel_requested"] = True
                if record["state"] == "queued":
                    record.update(state="cancelled", finished_at=time.time())
                    record["result"] = self.summary(record, "cancelled", "cancelled before execution")
        return self.update(job_id, change)

    def summary(self, record, state, error=None, execution=None):
        return dict(version=1, id=record["id"], state=state, error=error,
                    payload=record["payload"], submitted_at=record["submitted_at"],
                    started_at=record.get("started_at"), finished_at=time.time(),
                    result_path=str(self.directory(record["id"]) / "summary.json"),
                    execution=execution or {}, correctness="not_assessed")

    def finish(self, job_id, state, error=None, execution=None):
        if state not in TERMINAL:
            raise ValueError("invalid terminal state")
        def change(record):
            if record["state"] in TERMINAL:
                return
            if record["cancel_requested"]:
                final_state, final_error = "cancelled", "cancelled by request"
            else:
                final_state, final_error = state, error
            record.update(state=final_state, finished_at=time.time())
            record["result"] = self.summary(record, final_state, final_error, execution)
        record = self.update(job_id, change)
        atomic(self.directory(job_id) / "summary.json", record["result"])
        return record

    def meta(self, key, value=None):
        with self.connect() as db:
            if value is not None:
                db.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", (key, canonical(value).decode()))
            row = db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None
