#!/usr/bin/env python3
"""Execute commands or synchronize source in the configured Linux checkout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict[str, str]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError("Copy .remote.example.json to .remote.json and configure the remote first") from None
    except (OSError, ValueError):
        raise ValueError("Cannot read .remote.json as UTF-8 JSON") from None
    if not isinstance(config, dict) or set(config) != {"host", "repo"}:
        raise ValueError(".remote.json requires exactly host and repo")
    host, repo = config["host"], config["repo"]
    if not isinstance(host, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", host):
        raise ValueError("host must be an SSH alias from ~/.ssh/config")
    if (not isinstance(repo, str) or not repo.startswith("/")
            or any(c in repo for c in "\0\n\r")):
        raise ValueError("repo must be an absolute Linux path")
    return config


def ssh_command(config: dict[str, str], command: list[str]) -> list[str]:
    if not command or not command[0] or any("\0" in arg for arg in command):
        raise ValueError("exec requires a command without NUL bytes")
    # SSH passes one string to the remote shell; quote both cwd and every argument.
    activate = 'set -e; if [ -f .remote.env ]; then source .remote.env; fi; exec "$@"'
    remote = "cd " + shlex.quote(config["repo"]) + " && exec " + shlex.join(
        ["bash", "--noprofile", "--norc", "-c", activate, "ignis-remote", *command])
    return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
            config["host"], remote]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    execute = actions.add_parser("exec", help="Run in the existing checkout; does not sync source or retry")
    execute.add_argument("command", nargs=argparse.REMAINDER, help="-- command [arguments ...]")
    synchronize = actions.add_parser("sync", help="Sync committed and uncommitted source; preserve remote caches")
    synchronize.add_argument("--dry-run", action="store_true", help="Report compressed transfer size without applying")
    args = parser.parse_args(argv)
    command = args.command if args.action == "exec" else []
    if command[:1] == ["--"]:
        command = command[1:]
    if args.action == "exec" and not command:
        parser.error("exec requires -- command [arguments ...]")
    try:
        config = load_config(ROOT / ".remote.json")
        if args.action == "sync":
            from remote_sync import sync
            return sync(ROOT, config, ssh_command, args.dry_run)
        result = subprocess.run(ssh_command(config, command))
        return result.returncode if result.returncode >= 0 else 128 - result.returncode
    except ValueError as error:
        print(f"remote: {error}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as error:
        return error.returncode if error.returncode > 0 else 128 - error.returncode
    except OSError:
        print("remote: cannot launch SSH; check the local OpenSSH installation", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
