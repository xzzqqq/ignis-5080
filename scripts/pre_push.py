#!/usr/bin/env python3
"""Install or run the repository's fail-closed pre-push secret check."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

MARKER = "# ignis pre-push secret guard"
OID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
PRIVATE_FILES = {".remote.json", ".remote.env", ".remote.cargo.toml", "local.mk", ".env"}
PRIVATE_DIRS = {".toolchain", ".scratch", ".ssh"}


def git(root, *args, check=True):
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                            env=dict(os.environ, GIT_NO_REPLACE_OBJECTS="1"), timeout=60)
    if check and result.returncode:
        # Git errors can contain remote URLs with credentials; do not echo them.
        raise ValueError(f"git {args[0]} failed; push blocked (check repository state or fetch destination history)")
    return result


def config(root, key):
    return git(root, "config", "--get", key, check=False).stdout.decode().strip()


def scanner(root):
    path = config(root, "ignis.gitleaksBinary") or shutil.which("gitleaks")
    if not path or not Path(path).is_file() or not os.access(path, os.X_OK):
        raise ValueError("gitleaks scanner missing; install Gitleaks and rerun this script with install")
    return str(Path(path).resolve())


def install(root):
    binary = shutil.which("gitleaks")
    if not binary:
        raise ValueError("gitleaks scanner missing; install Gitleaks 8.30.1 or later first")
    hooks = git(root, "config", "--path", "--get", "core.hooksPath", check=False).stdout.decode().strip()
    hook = Path(hooks) / "pre-push" if hooks else Path(git(root, "rev-parse", "--git-path", "hooks/pre-push").stdout.decode().strip())
    if not hook.is_absolute():
        hook = root / hook
    if hook.is_symlink() or (hook.exists() and MARKER not in hook.read_text()):
        raise ValueError("existing pre-push hook preserved; integrate the secret check into it before installing")
    body = ('#!/bin/sh\n' + MARKER + '\nset -eu\n'
            'root=$(git rev-parse --show-toplevel)\n'
            'python=$(git config --get ignis.pythonBinary || printf \'%s\\n\' python3)\n'
            'exec "$python" "$root/scripts/pre_push.py" check "$@"\n')
    hook.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=hook.parent, delete=False) as out:
        out.write(body)
        temporary = Path(out.name)
    try:
        temporary.chmod(0o755)
        git(root, "config", "--local", "ignis.gitleaksBinary", str(Path(binary).resolve()))
        git(root, "config", "--local", "ignis.pythonBinary", sys.executable)
        temporary.replace(hook)
    finally:
        temporary.unlink(missing_ok=True)
    print("Installed pre-push secret check; existing core.hooksPath and other hooks preserved")


def peel(root, oid, metadata=None):
    while True:
        kind = git(root, "cat-file", "-t", oid).stdout.strip()
        if kind == b"commit":
            return oid
        if kind != b"tag":
            raise ValueError("only branches and tags pointing to commits are supported; push blocked")
        body = git(root, "cat-file", "tag", oid).stdout
        if metadata is not None:
            metadata["tag-" + oid] = body
        oid = body.split(b"\n", 1)[0].removeprefix(b"object ").decode()
        if not OID.fullmatch(oid):
            raise ValueError("invalid annotated tag target; push blocked")


def scope(root, location, lines):
    tips, bases, metadata = set(), set(), {}
    new_ref = False
    for line in lines:
        fields = line.split()
        if len(fields) != 4:
            raise ValueError("invalid pre-push input; push blocked")
        local_ref, local_oid, remote_ref, remote_oid = fields
        if not OID.fullmatch(local_oid) or not OID.fullmatch(remote_oid):
            raise ValueError("invalid object ID; push blocked")
        if (not remote_ref.startswith(("refs/heads/", "refs/tags/"))
                or (local_ref.startswith("refs/") and not local_ref.startswith(("refs/heads/", "refs/tags/")))):
            raise ValueError("internal ref push blocked; push an explicit branch or tag, never --mirror")
        if set(local_oid) == {"0"}:
            continue  # Deletion uploads no content.
        tips.add(peel(root, local_oid, metadata))
        if set(remote_oid) == {"0"}:
            new_ref = True
        else:
            bases.add(peel(root, remote_oid))
    if tips and git(root, "rev-parse", "--is-shallow-repository").stdout.strip() == b"true":
        raise ValueError("shallow history cannot be audited; fetch --unshallow before pushing")
    if new_ref:
        # Cached remote-tracking refs can be stale or refer to a different push URL.
        advertised = git(root, "ls-remote", "--refs", "--", location).stdout.decode().splitlines()
        for line in advertised:
            oid, ref = line.split("\t", 1)
            if OID.fullmatch(oid) and ref.startswith(("refs/heads/", "refs/tags/")):
                known = git(root, "rev-parse", "--verify", oid + "^{commit}", check=False)
                if known.returncode == 0:
                    bases.add(known.stdout.decode().strip())
        # An empty destination or no locally-known advertised commits means a full
        # reachable-history scan. Unknown remote objects never become exclusions.
    revisions = sorted(tips) + ["^" + oid for oid in sorted(bases)]
    commits = git(root, "rev-list", *revisions, "--").stdout.decode().splitlines() if tips else []
    return revisions, commits, metadata


def private_paths(root, commits):
    for commit in commits:
        names = git(root, "diff-tree", "--root", "-m", "--no-commit-id", "--name-only", "-r",
                    "--no-renames", "--diff-filter=ACMRT", "-z", commit).stdout.split(b"\0")
        for name in filter(None, names):
            path = os.fsdecode(name)
            parts = path.lower().split("/")
            if parts[-1] in PRIVATE_FILES or any(part in PRIVATE_DIRS for part in parts):
                raise ValueError(f"private path {path!r} in commit {commit[:12]}; remove it from outgoing history")


def scan(root, binary, directory, mode, arguments):
    report = directory / (mode + ".json")
    command = [binary, mode, "--config", str(directory / "rules.toml"),
               "--gitleaks-ignore-path", str(directory / "empty.ignore"), "--ignore-gitleaks-allow",
               "--redact=100", "--no-banner", "--no-color", "--max-decode-depth=2",
               "--max-archive-depth=2", "--report-format=json", "--report-path", str(report), *arguments]
    environment = dict(os.environ, GIT_NO_REPLACE_OBJECTS="1")
    environment.pop("GITLEAKS_CONFIG", None)
    environment.pop("GITLEAKS_CONFIG_TOML", None)
    result = subprocess.run(command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
                            capture_output=True, timeout=180)
    if result.returncode not in {0, 1} or not report.exists():
        raise ValueError(f"gitleaks scanner failed (exit {result.returncode}); push blocked")
    findings = json.loads(report.read_text())
    if not isinstance(findings, list) or (result.returncode != 0 and not findings):
        raise ValueError("gitleaks scanner did not complete successfully; push blocked")
    if findings:
        for item in findings:
            file = item["File"]
            if mode == "dir":
                file = Path(file).stem + " message"
            print(f"pre-push: {item['RuleID']} at {file!r}:{item['StartLine']} "
                  f"commit {item.get('Commit', '')[:12]}", file=sys.stderr)
        raise ValueError("possible credentials found; push blocked (secret values hidden)")


def check(root, location, lines):
    revisions, commits, metadata = scope(root, location, lines)
    private_paths(root, commits)
    if not commits and not metadata:
        print("pre-push: no new content; passed", file=sys.stderr)
        return
    binary = scanner(root)
    with tempfile.TemporaryDirectory(prefix="ignis-pre-push-") as temporary:
        directory = Path(temporary)
        (directory / "rules.toml").write_text("[extend]\nuseDefault = true\n")
        (directory / "empty.ignore").write_text("")
        if commits:
            scan(root, binary, directory, "git",
                 ["--log-opts=" + " ".join(["--full-history", "-m", *revisions]), str(root)])
        # git-mode scans patches; commit/tag messages need their own content scan.
        for commit in commits:
            metadata["commit-" + commit] = git(root, "cat-file", "commit", commit).stdout
        if metadata:
            folder = directory / "messages"
            folder.mkdir()
            for name, body in metadata.items():
                (folder / (name + ".txt")).write_bytes(body)
            scan(root, binary, directory, "dir", [str(folder)])
    print(f"pre-push: {len(commits)} outgoing commits and tag messages checked; passed", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["install", "check"])
    parser.add_argument("remote", nargs="?")
    parser.add_argument("location", nargs="?")
    args = parser.parse_args()
    try:
        root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel").stdout.decode().strip())
        if args.action == "install":
            install(root)
        else:
            if not args.location:
                raise ValueError("check must receive the pre-push remote arguments")
            check(root, args.location, sys.stdin.read().splitlines())
        return 0
    except subprocess.TimeoutExpired:
        print("pre-push: Git or scanner timed out; push blocked", file=sys.stderr)
        return 1
    except (OSError, ValueError) as error:
        print(f"pre-push: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
