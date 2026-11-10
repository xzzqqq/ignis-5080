"""Laptop transport. Receipts precede SSH; retry sends the same frozen bytes."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

from .schema import canonical, identifier, relative, validate
from .source import freeze, sha256
from .store import atomic

# The bootstrap only stages the tool and exact named payloads. It never checks
# out source or imports code from the configured remote working directory.
BOOTSTRAP = r'''
import hashlib, json, os, pathlib, shutil, subprocess, sys, tarfile, tempfile
root = pathlib.Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory(prefix="ignis-upload-") as temporary:
    temporary = pathlib.Path(temporary)
    with tarfile.open(fileobj=sys.stdin.buffer, mode="r|gz") as archive:
        seen = set()
        for member in archive:
            if member.name not in {"runtime.pyz", "payload.json", "source.bundle"} or not member.isfile() or member.name in seen:
                raise ValueError("invalid upload entry")
            seen.add(member.name)
            with (temporary / member.name).open("wb") as output:
                shutil.copyfileobj(archive.extractfile(member), output)
    data = (temporary / "runtime.pyz").read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != sys.argv[2]:
        raise ValueError("runtime hash mismatch")
    tools = root / ".scratch/experiment-tools"
    tools.mkdir(parents=True, exist_ok=True, mode=0o700)
    runtime = tools / (digest + ".pyz")
    if not runtime.exists():
        fd, name = tempfile.mkstemp(dir=tools)
        with os.fdopen(fd, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(name, 0o400)
        os.replace(name, runtime)
    elif hashlib.sha256(runtime.read_bytes()).hexdigest() != digest:
        raise ValueError("deployed runtime was modified")
    command = [sys.executable, str(runtime), "--root", str(root), *sys.argv[3:]]
    bundle = temporary / "source.bundle"
    if bundle.exists():
        command += ["--bundle", str(bundle)]
    with (temporary / "payload.json").open("rb") as payload:
        sys.exit(subprocess.run(command, stdin=payload).returncode)
'''


def package():
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        files = {"__main__.py": b"from experiments.host import entry\nraise SystemExit(entry())\n"}
        for path in sorted(Path(__file__).parent.glob("*.py")):
            if path.name != "client.py":
                files["experiments/" + path.name] = path.read_bytes()
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, (2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    return data.getvalue()


def upload(path, runtime, payload, bundle=None):
    with tarfile.open(path, "w:gz") as archive:
        for name, data in [("runtime.pyz", runtime), ("payload.json", canonical(payload))]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        if bundle:
            archive.add(bundle, arcname="source.bundle", recursive=False)


def call(config, ssh_command, runtime_hash, archive, arguments, output=None):
    command = ssh_command(config, ["python3", "-c", BOOTSTRAP, config["repo"], runtime_hash, *arguments])
    with Path(archive).open("rb") as stream:
        return subprocess.run(command, stdin=stream, stdout=output)


def unpack(path, destination):
    """Verify a portable fetch before publishing it; never follow tar links."""
    if destination.exists():
        raise ValueError("fetch destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        staging = Path(temporary) / "result"
        staging.mkdir()
        with tarfile.open(path, "r:gz") as archive:
            names = set()
            for member in archive:
                relative(member.name)
                if not member.isfile() or member.name in names:
                    raise ValueError("fetch contains links, duplicate paths or non-files")
                names.add(member.name)
                target = staging / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("wb") as output:
                    shutil.copyfileobj(archive.extractfile(member), output)
        manifest = json.loads((staging / "manifest.json").read_text())
        if set(manifest) != names - {"manifest.json"}:
            raise ValueError("fetch manifest has missing or extra files")
        for name, expected in manifest.items():
            target = staging / name
            if sha256(target) != expected["sha256"] or target.stat().st_size != expected["bytes"]:
                raise ValueError("fetch hash mismatch: " + name)
        os.rename(staging, destination)


def main(root, config, ssh_command, argv):
    parser = argparse.ArgumentParser(prog="remote.py experiment", description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    install = actions.add_parser("install", help="Install/enable the immutable systemd user worker")
    install.add_argument("--idle-mib", type=int, default=256)
    for name in ("start", "stop", "status"):
        actions.add_parser(name)
    submit = actions.add_parser("submit", help="Freeze this worktree and enqueue its request")
    submit.add_argument("request", type=Path)
    submit.add_argument("--commit", help="Use only this committed revision instead of working files")
    for name in ("retry", "result", "wait", "cancel", "fetch", "prune"):
        action = actions.add_parser(name)
        action.add_argument("id")
        if name == "wait":
            action.add_argument("--timeout", type=float, default=60)
        if name == "fetch":
            action.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if getattr(args, "id", None):
        identifier(args.id)
    receipts = root / ".scratch/experiment-submissions"
    runtime = package()
    runtime_hash = hashlib.sha256(runtime).hexdigest()
    with tempfile.TemporaryDirectory(prefix="ignis-client-") as temporary:
        temporary = Path(temporary)
        archive = temporary / "upload.tar.gz"
        upload(archive, runtime, {})
        if args.action == "submit":
            request = validate(json.loads(args.request.read_text()))
            receipt = receipts / request["id"]
            if receipt.exists():
                raise ValueError("local submission id already frozen; use experiment retry " + request["id"])
            with (temporary / "probe.json").open("wb") as handle:
                result = call(config, ssh_command, runtime_hash, archive, ["probe"], handle)
            if result.returncode:
                return result.returncode
            probe = json.loads((temporary / "probe.json").read_text())
            with freeze(root, probe["head"], args.commit) as (source, bundle):
                payload = dict(request=request, source=source, runtime_sha256=runtime_hash)
                receipt.mkdir(parents=True, exist_ok=False, mode=0o700)
                atomic(receipt / "receipt.json", dict(payload=payload, config=config, runtime_sha256=runtime_hash))
                upload(receipt / "upload.tar.gz", runtime, payload, bundle)
                # All retry bytes exist before the first side-effecting SSH call.
                with (receipt / "upload.tar.gz").open("rb") as handle:
                    os.fsync(handle.fileno())
                atomic(receipt / "ready.json", dict(ready=True))
            archive = receipt / "upload.tar.gz"
            arguments = ["submit"]
        elif args.action == "retry":
            receipt = receipts / args.id
            if not (receipt / "ready.json").exists():
                raise ValueError("no complete local submission receipt for this id")
            saved = json.loads((receipt / "receipt.json").read_text())
            if saved["config"] != config:
                raise ValueError("remote config changed since submission; restore it before retry")
            runtime_hash = saved["runtime_sha256"]
            archive = receipt / "upload.tar.gz"
            arguments = ["submit"]
        else:
            arguments = [args.action]
            if getattr(args, "id", None):
                arguments += [args.id]
            if args.action == "wait":
                arguments += ["--timeout", str(args.timeout)]
            if args.action == "install":
                arguments += ["--idle-mib", str(args.idle_mib)]
        if args.action == "fetch":
            destination = (args.out or root / ".scratch/experiment-results" / args.id).resolve()
            if destination.exists():
                raise ValueError("fetch destination already exists")
            downloaded = temporary / "result.tar.gz"
            with downloaded.open("wb") as handle:
                result = call(config, ssh_command, runtime_hash, archive, arguments, handle)
            if result.returncode == 0:
                unpack(downloaded, destination)
                print(json.dumps(dict(id=args.id, result_path=str(destination))))
            return result.returncode
        if args.action in {"submit", "retry"}:
            with (receipt / "response.json").open("wb") as handle:
                result = call(config, ssh_command, runtime_hash, archive, arguments, handle)
            sys.stdout.write((receipt / "response.json").read_text())
            if result.returncode:
                print(f"Frozen receipt: {receipt}; retry this id after inspecting remote status", file=sys.stderr)
            return result.returncode
        return call(config, ssh_command, runtime_hash, archive, arguments).returncode
