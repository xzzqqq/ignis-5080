"""FIFO scheduling and crash reconciliation. Never replay an uncertain attempt."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from . import TERMINAL
from . import systemd
from .source import sha256
from .runner import artifact
from .store import Store, atomic

GPU_LOCK = Path("/tmp/ignis-gpu-profile.lock")


def gpu_status(root, idle_mib):
    status = subprocess.run(["make", "gpu-status"], cwd=root, capture_output=True, text=True, timeout=15)
    if status.returncode:
        raise ValueError("make gpu-status failed: " + status.stderr.strip())
    def query(option):
        return subprocess.check_output(["nvidia-smi", option, "--format=csv,noheader,nounits"],
                                       text=True, timeout=15).strip()
    cards = query("--query-gpu=name,memory.used,uuid,driver_version").splitlines()
    if len(cards) != 1 or "RTX 5080" not in cards[0]:
        raise ValueError("experiments require exactly one RTX 5080")
    apps = query("--query-compute-apps=pid,process_name,used_memory")
    if apps or float(cards[0].split(",")[1]) > idle_mib:
        raise ValueError("GPU busy: " + (apps or cards[0]))
    return dict(card=cards[0], status=status.stdout, checked_at=time.time())


def load_execution(directory):
    path = directory / "execution.json"
    return json.loads(path.read_text()) if path.exists() else {}


def reconcile(store, record, idle_mib, lease=None):
    """A running unit survives worker/SSH loss; finalization follows cgroup cleanup."""
    directory = store.directory(record["id"])
    unit = systemd.inspect(record["unit"])
    active = unit.get("ActiveState") in {"active", "activating", "deactivating"}
    done_path = directory / "done.json"
    if active and unit.get("SubState") != "exited" and not record["cancel_requested"]:
        return False
    # Capture systemd's timeout/exit diagnosis before stop/reset-failed erases it.
    completion = record.get("completion")
    if completion is None:
        completion = dict(unit=unit, done=json.loads(done_path.read_text()) if done_path.exists() else None)
        store.update(record["id"], lambda row: row.update(completion=completion))
    diagnosis = completion["unit"].get("Result")
    systemd.stop(record["unit"])
    if lease is None:
        lease = GPU_LOCK.open("a")
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            lease.close()
            raise
        own_lease = True
    else:
        own_lease = False
    try:
        released = gpu_status(store.root, idle_mib)
        if completion["done"] is not None:
            done = completion["done"]
        else:
            done = dict(state="timed_out" if diagnosis == "timeout" else "interrupted",
                        error="unit ended without a complete result: " + str(diagnosis),
                        execution=load_execution(directory))
        # A timeout can kill a wrapper after it wrote done but before it exited.
        if diagnosis == "timeout":
            done.update(state="timed_out", error="experiment deadline exceeded")
        if done["state"] == "succeeded":
            try:
                for step in done["execution"]["steps"]:
                    for expected in step["artifacts"]:
                        if artifact(directory / "output", expected["path"]) != expected:
                            raise ValueError("artifact changed before cleanup: " + expected["path"])
            except ValueError as exception:
                done.update(state="failed", error=str(exception))
        done["execution"]["unit_result"] = completion["unit"]
        done["execution"]["gpu_released"] = released
        store.finish(record["id"], done["state"], done["error"], done["execution"])
        return True
    finally:
        if own_lease:
            lease.close()


def serve(root, runtime, idle_mib=256, poll_s=1):
    store = Store(root)
    stopping = False
    def stop_signal(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    # A second worker cannot consume even if manually started outside systemd.
    with (store.path / "worker.lock").open("a") as worker_lock:
        fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lease = None
        try:
            while True:
                heartbeat = dict(pid=os.getpid(), runtime_sha256=sha256(runtime), at=time.time(),
                                 stopping=stopping, blocked=None)
                store.meta("worker", heartbeat)
                try:
                    jobs = store.jobs()
                    running = next((job for job in jobs if job["state"] == "running"), None)
                    if running:
                        if stopping:
                            # Normal stop cancels the active unit; queued jobs remain.
                            store.cancel(running["id"])
                            running = store.get(running["id"])
                        if reconcile(store, running, idle_mib, lease):
                            if lease:
                                lease.close()
                                lease = None
                    elif stopping:
                        return 0
                    else:
                        queued = next((job for job in jobs if job["state"] == "queued"), None)
                        if queued:
                            if lease is None:
                                lease = GPU_LOCK.open("a")
                                try:
                                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                except BaseException:
                                    lease.close()
                                    lease = None
                                    raise
                            preflight = gpu_status(root, idle_mib)
                            job_id = queued["id"]
                            with systemd.listener(root, job_id) as channel:
                                record = store.claim(job_id, systemd.unit_name(root, job_id))
                                atomic(store.directory(job_id) / "preflight.json", preflight)
                                systemd.launch(root, runtime, record)
                                systemd.send_lease(channel, lease.fileno())
                        elif lease:
                            lease.close()
                            lease = None
                except Exception as exception:
                    # Retain the running record and lease until cleanup is proven.
                    # A failed launch is reconciled on the next iteration, not retried.
                    heartbeat["blocked"] = f"{type(exception).__name__}: {exception}"
                    store.meta("worker", heartbeat)
                    if lease and not any(job["state"] == "running" for job in store.jobs()):
                        lease.close()
                        lease = None
                time.sleep(poll_s)
        finally:
            if lease:
                lease.close()
