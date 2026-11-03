"""Incremental Git bundles for a Laptop-owned checkout; no build/model transfer."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid


def git(root, *args, input=None, env=None, check=True):
    result = subprocess.run(["git", "-C", str(root), *args], input=input,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    if check and result.returncode:
        raise ValueError(result.stderr.decode(errors="replace").strip())
    return result


def oid(root, revision):
    result = git(root, "rev-parse", "--verify", revision, check=False)
    return result.stdout.decode().strip() if result.returncode == 0 else None


def applied_ref(root):
    name = hashlib.sha256(os.fsencode(root.resolve())).hexdigest()[:16]
    return "refs/ignis-sync/applied/" + name


def probe(root):
    head = oid(root, "HEAD")
    if head is None:
        raise ValueError("remote checkout has no HEAD")
    snapshot = oid(root, applied_ref(root))
    if snapshot and oid(root, snapshot + "^") != head:
        raise ValueError("remote HEAD changed independently since the last sync")
    base = snapshot or head
    for args in [("diff", "--quiet", base, "--"), ("diff", "--cached", "--quiet", base, "--")]:
        if git(root, *args, check=False).returncode:
            raise ValueError("remote has independent tracked/index changes; preserve or resolve them before sync")
    if git(root, "ls-files", "--others", "--exclude-standard", "-z").stdout:
        raise ValueError("remote has untracked files; preserve or resolve them before sync")
    return {"head": head, "snapshot": snapshot}


def private_path(name):
    return (name in {".remote.json", ".remote.env", ".remote.cargo.toml", "local.mk"}
            or name.split("/", 1)[0] in {".git", ".scratch", ".toolchain", "models", "target", "node_modules"}
            or name.startswith("kernel/build/") or name.startswith("web/node_modules/"))


@contextmanager
def prepare(root, state):
    """Snapshot the working tree through a temporary index, leaving staging untouched."""
    head = oid(root, "HEAD")
    if not oid(root, state["head"] + "^{commit}"):
        raise ValueError("remote HEAD is unknown locally; sync requires the same repository history")
    if not state["snapshot"] and git(root, "merge-base", "--is-ancestor", state["head"], head,
                                      check=False).returncode:
        raise ValueError("initial remote HEAD is not an ancestor of the Laptop HEAD")
    ref = "refs/ignis-sync/export/" + uuid.uuid4().hex
    try:
        with tempfile.TemporaryDirectory(prefix="ignis-sync-") as directory:
            directory = Path(directory)
            index = Path(git(root, "rev-parse", "--git-path", "index").stdout.decode().strip())
            if not index.is_absolute():
                index = root / index
            env = dict(os.environ, GIT_INDEX_FILE=str(directory / "index"))
            if index.exists():
                shutil.copyfile(index, directory / "index")
            else:
                git(root, "read-tree", head, env=env)
            git(root, "add", "--all", "--", ".", env=env)
            entries = git(root, "ls-files", "--stage", "-z", env=env).stdout.split(b"\0")
            for entry in filter(None, entries):
                meta, name = entry.split(b"\t", 1)
                if private_path(os.fsdecode(name)):
                    raise ValueError("snapshot contains a tracked private/build path: " + os.fsdecode(name))
                if meta.startswith(b"160000 "):
                    raise ValueError("sync does not support submodules")
            tree = git(root, "write-tree", env=env).stdout.decode().strip()
            previous = state["snapshot"] or state["head"]
            if (state["snapshot"] and oid(root, previous + "^") == head
                    and oid(root, previous + "^{tree}") == tree):
                yield {"bundle": None, "bytes": 0, "files": 0, "snapshot": previous, "head": head, "ref": ref}
                return
            # A deterministic internal commit identifies this tree and its source HEAD.
            # The prior snapshot is a parent too, so bundles can omit its unchanged
            # uncommitted blobs. No user branch or index is committed or changed.
            env.update(GIT_AUTHOR_NAME="Ignis sync", GIT_AUTHOR_EMAIL="sync@localhost",
                       GIT_COMMITTER_NAME="Ignis sync", GIT_COMMITTER_EMAIL="sync@localhost",
                       GIT_AUTHOR_DATE="@0 +0000", GIT_COMMITTER_DATE="@0 +0000")
            parents = ["-p", head]
            if state["snapshot"] and oid(root, previous + "^{commit}"):
                parents += ["-p", previous]
            snapshot = git(root, "commit-tree", tree, *parents,
                           input=b"Ignis working tree snapshot\n", env=env).stdout.decode().strip()
            git(root, "update-ref", ref, snapshot)
            bundle = directory / "source.bundle"
            exclusions = ["^" + state["head"]]
            if state["snapshot"] and oid(root, state["snapshot"] + "^{commit}"):
                exclusions.append("^" + state["snapshot"])
            git(root, "bundle", "create", str(bundle), ref, *exclusions)
            files = None
            if oid(root, previous + "^{commit}"):
                files = len(list(filter(None, git(root, "diff", "--name-only", "-z", previous, snapshot).stdout.split(b"\0"))))
            yield {"bundle": str(bundle), "bytes": bundle.stat().st_size, "files": files,
                   "snapshot": snapshot, "head": head, "ref": ref}
    finally:
        git(root, "update-ref", "-d", ref, check=False)


def receive(root, expected, ref, snapshot, stream):
    """Import the bundle, recheck ownership, then update only Git-managed sources."""
    gitdir = Path(git(root, "rev-parse", "--absolute-git-dir").stdout.decode().strip())
    with (gitdir / "ignis-sync.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if probe(root) != expected:
            raise ValueError("remote state changed during sync; nothing applied")
        incoming = "refs/ignis-sync/incoming/" + uuid.uuid4().hex
        try:
            with tempfile.TemporaryDirectory(prefix="ignis-sync-", dir=gitdir) as directory:
                bundle = Path(directory) / "source.bundle"
                with bundle.open("wb") as output:
                    shutil.copyfileobj(stream, output)
                git(root, "bundle", "verify", str(bundle))
                git(root, "fetch", "--quiet", "--no-tags", "--no-write-fetch-head",
                    str(bundle), ref + ":" + incoming)
                if oid(root, incoming) != snapshot:
                    raise ValueError("received snapshot differs from the requested source")
                source_head = oid(root, snapshot + "^")
                base = expected["snapshot"] or expected["head"]
                tracked = set(git(root, "ls-files", "-z").stdout.split(b"\0"))
                added = git(root, "diff", "--no-renames", "--name-only", "--diff-filter=A", "-z", base, snapshot).stdout
                for name in filter(None, added.split(b"\0")):
                    path = root / os.fsdecode(name)
                    if path.exists() or path.is_symlink():
                        raise ValueError("sync would overwrite an existing remote file: " + os.fsdecode(name))
                    for parent in path.parents:
                        if parent == root:
                            break
                        relative = parent.relative_to(root)
                        if (parent.is_symlink() or parent.is_file()) and os.fsencode(relative) not in tracked:
                            raise ValueError("sync would overwrite an existing remote path: " + str(relative))
                if probe(root) != expected:
                    raise ValueError("remote state changed during transfer; nothing applied")
                git(root, "read-tree", "-u", "-m", base, snapshot)
                git(root, "update-ref", "HEAD", source_head, expected["head"])
                git(root, "update-ref", applied_ref(root), snapshot)
                return {"head": source_head, "snapshot": snapshot}
        finally:
            git(root, "update-ref", "-d", incoming, check=False)


def sync(root, config, ssh_command, dry_run=False):
    # Send the receiver itself, so a checkout predating sync can be updated too.
    receiver = Path(__file__).read_text()
    command = ssh_command(config, ["python3", "-c", receiver, "probe"])
    result = subprocess.run(command, stdout=subprocess.PIPE, check=True)
    state = json.loads(result.stdout)
    with prepare(root, state) as prepared:
        files = prepared["files"]
        print(f"sync: {files if files is not None else 'unknown'} changed source files; "
              f"{prepared['bytes']} bundle bytes; HEAD {prepared['head'][:12]}", flush=True)
        if dry_run or prepared["bundle"] is None:
            print("sync: dry run; nothing applied" if dry_run else "sync: already current")
            return 0
        command = ssh_command(config, ["python3", "-c", receiver, "receive", json.dumps(state),
                                       prepared["ref"], prepared["snapshot"]])
        with Path(prepared["bundle"]).open("rb") as stream:
            subprocess.run(command, stdin=stream, check=True)
        return 0


if __name__ == "__main__":
    try:
        if sys.argv[1] == "probe":
            print(json.dumps(probe(Path.cwd())))
        elif sys.argv[1] == "receive":
            state = receive(Path.cwd(), json.loads(sys.argv[2]), sys.argv[3], sys.argv[4], sys.stdin.buffer)
            print("sync: applied " + state["snapshot"][:12])
        else:
            raise ValueError("unknown sync receiver action")
    except (ValueError, OSError) as error:
        print(f"sync: {error}", file=sys.stderr)
        sys.exit(1)
