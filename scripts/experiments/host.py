"""Remote control plane, deployed as a content-addressed Python zip application."""
import argparse
import fcntl
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

from . import PROTOCOL, TERMINAL, systemd
from .schema import canonical, validate
from .source import common_root, git, receive, sha256
from .store import Store, atomic


def runtime_path():
    return Path(sys.argv[0]).resolve()


def quote_unit(value):
    # systemd unit syntax is not shell syntax: escape its specifiers and quotes.
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'


def install(store, runtime, idle_mib):
    if idle_mib < 0:
        raise ValueError("idle-mib must be nonnegative")
    # Keep the user manager alive after the final SSH session closes, and start
    # enabled units at boot. No sudo or interactive authentication is used.
    linger = subprocess.run(["loginctl", "show-user", str(os.getuid()), "-p", "Linger", "--value"],
                            text=True, capture_output=True, timeout=15)
    if linger.returncode or linger.stdout.strip() != "yes":
        enabled = subprocess.run(["loginctl", "--no-ask-password", "enable-linger"],
                                 text=True, capture_output=True, timeout=15)
        if enabled.returncode:
            raise ValueError("user lingering is required for disconnect/reboot survival: " + enabled.stderr.strip())
    name = systemd.service_name(store.root)
    unit = Path.home() / ".config/systemd/user" / name
    text = ("[Unit]\nDescription=Ignis single GPU experiments\n"
            "[Service]\nType=exec\nExecStart=/usr/bin/python3 " + quote_unit(runtime)
            + " --root " + quote_unit(store.root) + f" serve --idle-mib {idle_mib}\n"
            "Restart=always\nRestartSec=2\nKillMode=control-group\nTimeoutStopSec=30\n"
            "UMask=0077\n[Install]\nWantedBy=default.target\n")
    if unit.exists() and unit.read_text() != text:
        if any(job["state"] not in TERMINAL for job in store.jobs()):
            raise ValueError("drain or cancel pending jobs before upgrading the worker")
        systemd.systemctl("stop", name)
    unit.parent.mkdir(parents=True, exist_ok=True)
    temporary = unit.with_suffix(".tmp")
    temporary.write_text(text)
    os.replace(temporary, unit)
    systemd.systemctl("daemon-reload")
    systemd.systemctl("enable", "--now", name)
    expected = sha256(runtime)
    for _ in range(100):
        worker = store.meta("worker")
        if worker and worker["runtime_sha256"] == expected and time.time() - worker["at"] < 5:
            return dict(service=name, runtime_sha256=expected, protocol=PROTOCOL)
        time.sleep(.1)
    raise ValueError("worker did not become ready; inspect systemctl --user status " + name)


def status(store):
    return dict(protocol=PROTOCOL, service=systemd.inspect(systemd.service_name(store.root)),
                worker=store.meta("worker"), jobs=store.jobs(), root=str(store.path))


def submit(store, payload, bundle, runtime):
    validate(payload["request"])
    if payload["runtime_sha256"] != sha256(runtime):
        raise ValueError("client/runtime mismatch")
    # Retries remain readable after an upgrade, even with the worker stopped.
    with (store.path / "submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        job_id = payload["request"]["id"]
        existing = next((job for job in store.jobs() if job["id"] == job_id), None)
        if existing:
            return store.submit(payload)
        state = status(store)
        worker = state["worker"]
        if (state["service"].get("ActiveState") != "active" or not worker
                or worker["runtime_sha256"] != payload["runtime_sha256"]
                or worker["stopping"] or time.time() - worker["at"] > 60):
            raise ValueError("matching worker is not ready; run experiment install/start, then retry")
        receive(store.root, payload["source"], bundle)
        directory = store.directory(job_id)
        if directory.exists():
            # The prior SSH process may have died after fsync/rename, before INSERT.
            if json.loads((directory / "request.json").read_text()) != payload:
                raise ValueError("id belongs to a different interrupted submission")
        else:
            directory.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=directory.parent) as temporary:
                staging = Path(temporary) / "job"
                (staging / "config").mkdir(parents=True)
                for name in (".remote.env", ".remote.cargo.toml"):
                    source = store.root / name
                    if source.exists():
                        shutil.copyfile(source, staging / "config" / name)
                        with (staging / "config" / name).open("rb") as handle:
                            os.fsync(handle.fileno())
                atomic(staging / "request.json", payload)
                os.rename(staging, directory)
                fd = os.open(directory.parent, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
        return store.submit(payload)


def export(store, job_id):
    record = store.get(job_id)
    if record["state"] not in TERMINAL:
        raise ValueError("fetch requires a terminal result")
    directory = store.directory(job_id)
    atomic(directory / "summary.json", record["result"])
    paths = [path for path in directory.iterdir()
             if path.is_file() and not path.is_symlink() and path.suffix in {".json", ".log"}]
    output = directory / "output"
    if output.is_symlink():
        raise ValueError("refusing symlinked output directory in fetch")
    if output.exists():
        for path in output.rglob("*"):
            if path.is_symlink() or not path.resolve().is_relative_to(output.resolve()):
                raise ValueError("refusing escaping output in fetch")
            if path.is_file():
                paths.append(path)
    manifest = {str(path.relative_to(directory)): dict(sha256=sha256(path), bytes=path.stat().st_size)
                for path in paths}
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz") as archive:
        data = canonical(manifest)
        info = tarfile.TarInfo("manifest.json")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
        for path in paths:
            archive.add(path, arcname=str(path.relative_to(directory)), recursive=False)


def prune(store, job_id):
    record = store.get(job_id)
    if record["state"] not in TERMINAL:
        raise ValueError("prune requires a terminal experiment")
    directory = store.directory(job_id)
    worktree = directory / "worktree"
    if worktree.exists():
        git(store.root, "worktree", "remove", "--force", str(worktree))
    for name in ("tmp", "cache"):
        shutil.rmtree(directory / name, ignore_errors=True)
    return dict(id=job_id, pruned=True, results_retained=str(directory))


def main(argv=None):
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("action", choices=["install", "start", "stop", "status", "probe", "submit",
                                          "result", "wait", "cancel", "fetch", "prune", "serve", "run"])
    parser.add_argument("id", nargs="?")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--idle-mib", type=int, default=256)
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout < 0:
        raise ValueError("wait timeout must be finite and nonnegative")
    store = Store(common_root(args.root))
    runtime = runtime_path()
    action = args.action
    if action == "serve":
        from .worker import serve
        return serve(store.root, runtime, args.idle_mib)
    if action == "run":
        from .runner import run
        return run(store.root, args.id)
    code = 0
    if action == "install":
        # Admission and upgrades share a lock: no request can slip between the
        # empty-queue check and replacing the running tool version.
        with (store.path / "submit.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result = install(store, runtime, args.idle_mib)
    elif action in {"start", "stop"}:
        systemd.systemctl(action, systemd.service_name(store.root))
        result = status(store)
    elif action == "probe":
        result = dict(head=git(store.root, "rev-parse", "HEAD"), protocol=PROTOCOL)
    elif action == "submit":
        result = submit(store, json.load(sys.stdin), args.bundle, runtime)
    elif action == "status":
        result = status(store)
    elif action == "cancel":
        result = store.cancel(args.id)
        if result["state"] == "running":
            systemd.systemctl("start", systemd.service_name(store.root))
    elif action in {"result", "wait"}:
        deadline = time.monotonic() + args.timeout
        while True:
            result = store.get(args.id)
            if result["state"] in TERMINAL:
                result = result["result"]
                code = 0 if result["state"] == "succeeded" else 1
                break
            if action == "result" or time.monotonic() >= deadline:
                from .worker import load_execution
                result["execution"] = load_execution(store.directory(args.id))
                code = 3
                break
            health = status(store)
            heartbeat = health["worker"]
            if (health["service"].get("ActiveState") != "active" or not heartbeat
                    or time.time() - heartbeat["at"] > 60 or heartbeat["blocked"]):
                result = dict(job=result, service=health["service"], worker=heartbeat)
                code = 4
                break
            time.sleep(.5)
    elif action == "fetch":
        export(store, args.id)
        return 0
    else:
        result = prune(store, args.id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return code


def entry():
    try:
        return main()
    except (ValueError, OSError) as error:
        print("experiment: " + str(error), file=sys.stderr)
        return 2
