"""The small, versioned request contract. Commands run as the trusted SSH user."""
import hashlib
import json
import math
from pathlib import PurePosixPath
import re

from . import PROTOCOL


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
        raise ValueError("id must be 1..64 ASCII letters, digits, underscores or hyphens")
    return value


def relative(value):
    if (not isinstance(value, str) or not value or "\0" in value
            or PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts
            or str(PurePosixPath(value)) != value or value == "."):
        raise ValueError("artifact must be a normalized relative file path")
    return value


def validate(request):
    if not isinstance(request, dict):
        raise ValueError("request must be an object")
    allowed = {"version", "id", "task", "question", "timeout_s", "steps", "env", "inputs", "config"}
    if set(request) - allowed or type(request.get("version")) is not int or request["version"] != PROTOCOL:
        raise ValueError("unknown request fields or unsupported version")
    identifier(request.get("id"))
    for key in ("task", "question"):
        if not isinstance(request.get(key), str) or not request[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    timeout = request.get("timeout_s")
    if isinstance(timeout, bool) or not isinstance(timeout, (float, int)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout_s must be finite and positive")
    steps = request.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("steps must be a nonempty array")
    names = set()
    for step in steps:
        if not isinstance(step, dict) or set(step) - {"name", "argv", "artifacts"}:
            raise ValueError("step accepts name, argv and artifacts only")
        name = identifier(step.get("name"))
        if name in names:
            raise ValueError("step names must be unique")
        names.add(name)
        argv = step.get("argv")
        if (not isinstance(argv, list) or not argv or not argv[0]
                or any(not isinstance(a, str) or "\0" in a for a in argv)):
            raise ValueError("argv must be a nonempty string array without NUL")
        artifacts = step.get("artifacts", [])
        if not isinstance(artifacts, list):
            raise ValueError("artifacts must be an array")
        for path in artifacts:
            relative(path)
    env = request.get("env", {})
    reserved = {"HOME", "CARGO_TARGET_DIR", "TMPDIR", "XDG_CACHE_HOME", "CUDA_CACHE_PATH"}
    if not isinstance(env, dict):
        raise ValueError("env must be an object")
    for key, value in env.items():
        if (not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or key in reserved
                or key.startswith("IGNIS_EXPERIMENT_") or key == "IGNIS_GPU_LEASE_FD"
                or not isinstance(value, str) or "\0" in value):
            raise ValueError("invalid or isolation-reserved environment variable")
    inputs = request.get("inputs", [])
    if not isinstance(inputs, list):
        raise ValueError("inputs must be an array")
    for item in inputs:
        if (not isinstance(item, dict) or set(item) != {"path", "sha256"}
                or not isinstance(item["path"], str) or not item["path"].startswith("/")
                or "\0" in item["path"] or not isinstance(item.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])):
            raise ValueError("inputs require an absolute remote path and sha256")
    if not isinstance(request.get("config", {}), dict):
        raise ValueError("config must be an object (descriptive; argv/env control execution)")
    canonical(request)
    return request
