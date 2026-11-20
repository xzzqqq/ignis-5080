#!/usr/bin/env python3
"""Compare fixed-history span_logits BF16 dumps; no CUDA or engine imports.

Manifest (paths relative to its location)::

  {"vocab":248320,"reference":"bf16","windows":[
    {"id":"code","tokens_file":"windows.json","window_index":0,
     "start":3072,"rows":1024,"logits":{
       "bf16":"bf16/w000.bin","hq":"hq/w000.bin","nvfp4":"nvfp4/w000.bin"}}]}

A window may embed ``tokens`` instead. Row i predicts tokens[start+i+1]; the
last unknown next token is excluded from NLL. Every arm must use the same
artifact, token history, chunk geometry and dump range. Results describe
incremental KV error against the reference, not task quality or a pass gate.
NumPy is optional but strongly recommended for full-vocabulary dumps. The
stdlib implementation also supports small fixtures and the unit tests.
"""

import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import struct
import sys

try:
    import numpy as np
except ImportError:
    np = None


def row_scores(reference, candidate, target=None):
    """KL(reference||candidate), fixed-history agreement and next-token NLL."""
    if not len(reference) or len(reference) != len(candidate):
        raise ValueError("equal nonempty logit rows required")
    if np is not None:
        r = np.asarray(reference, dtype=np.float64)
        c = np.asarray(candidate, dtype=np.float64)
        if r.ndim != 1 or c.shape != r.shape:
            raise ValueError("one-dimensional logit rows required")
        if not np.isfinite(r).all() or not np.isfinite(c).all():
            raise ValueError("nonfinite logits")
        dr, dc = r - r.max(), c - c.max()
        lr = dr - np.log(np.exp(dr).sum(dtype=np.float64))
        lc = dc - np.log(np.exp(dc).sum(dtype=np.float64))
        pick, other = int(np.argmax(r)), int(np.argmax(c))
        pr, pc = np.exp(lr), np.exp(lc)
        kl = float(np.sum(pr * (lr - lc), dtype=np.float64))
        entropy_r = float(-np.sum(pr * lr, dtype=np.float64))
        entropy_c = float(-np.sum(pc * lc, dtype=np.float64))
        confidence_r, confidence_c = float(pr[pick]), float(pc[other])
        drift = float(np.max(np.abs(r - c)))
    else:
        r, c = list(reference), list(candidate)
        if not all(math.isfinite(x) for x in r + c):
            raise ValueError("nonfinite logits")

        def log_softmax(row):
            top = max(row)
            d = [x - top for x in row]
            z = math.log(math.fsum(math.exp(x) for x in d))
            return [x - z for x in d]

        lr, lc = log_softmax(r), log_softmax(c)
        pick = max(range(len(r)), key=r.__getitem__)
        other = max(range(len(c)), key=c.__getitem__)
        pr, pc = [math.exp(x) for x in lr], [math.exp(x) for x in lc]
        kl = math.fsum(p * (a - b) for p, a, b in zip(pr, lr, lc))
        entropy_r = -math.fsum(p * a for p, a in zip(pr, lr))
        entropy_c = -math.fsum(p * a for p, a in zip(pc, lc))
        confidence_r, confidence_c = pr[pick], pc[other]
        drift = max(abs(a - b) for a, b in zip(r, c))
    result = {
        "kl_ref_to_arm": kl,
        "top1_agreement": float(pick == other),
        "reference_pick_gap": float(r[pick] - r[other]),
        "max_abs_logit_drift": drift,
        "entropy": entropy_c,
        "reference_entropy": entropy_r,
        "max_probability": confidence_c,
        "reference_max_probability": confidence_r,
    }
    if target is not None:
        if not 0 <= target < len(r):
            raise ValueError("target outside vocabulary")
        result.update(nll=float(-lc[target]), reference_nll=float(-lr[target]),
                      next_token_top1_correct=float(other == target),
                      reference_next_token_top1_correct=float(pick == target))
        if other != target:
            result.update(entropy_on_wrong_next_token=entropy_c,
                          max_probability_on_wrong_next_token=confidence_c)
    return result


def percentile(ordered, q):
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def aggregate(rows):
    if not rows:
        raise ValueError("empty scored rows")
    result = {}
    for key in sorted(set().union(*(row.keys() for row in rows))):
        values = sorted(row[key] for row in rows if key in row)
        result[key] = {
            "mean": statistics.fmean(values), "median": statistics.median(values),
            "p99": percentile(values, .99), "max": values[-1], "count": len(values),
        }
    return result


def paired_windows(records, candidate, control, seed=0, repeats=2000):
    """Bootstrap paired window means, not correlated tokens as independent data."""
    result = {}
    rng = random.Random(seed)
    for key in ("kl_ref_to_arm", "nll"):
        deltas = [r["arms"][candidate][key]["mean"] - r["arms"][control][key]["mean"]
                  for r in records if key in r["arms"][candidate] and key in r["arms"][control]]
        if not deltas:
            continue
        interval = None
        if len(deltas) > 1:
            draws = sorted(statistics.fmean(rng.choices(deltas, k=len(deltas)))
                           for _ in range(repeats))
            interval = [percentile(draws, .025), percentile(draws, .975)]
        result[key] = {
            "candidate_minus_control_window_mean": statistics.fmean(deltas),
            "windows": len(deltas), "per_window": deltas,
            "window_bootstrap_95pct": interval,
        }
    return result


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_row(source, vocab):
    bits = array("H")
    bits.frombytes(source.read(vocab * 2))
    if sys.byteorder != "little":
        bits.byteswap()
    if np is not None:
        return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)
    return [struct.unpack("<f", struct.pack("<I", value << 16))[0] for value in bits]


def run(manifest_path):
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    vocab = int(manifest["vocab"])
    reference = manifest.get("reference", "bf16")
    if vocab <= 0 or not manifest["windows"]:
        raise ValueError("positive vocabulary and nonempty windows required")
    windows, pooled, hashes = [], {}, {}
    expected_arms = None
    for window in manifest["windows"]:
        tokens = window.get("tokens")
        if tokens is None:
            token_windows = json.loads((path.parent / window["tokens_file"]).read_text())
            tokens = token_windows[int(window["window_index"])]
        start, count = int(window["start"]), int(window["rows"])
        if start < 0 or count <= 0 or start + count > len(tokens):
            raise ValueError("bad scored token range")
        arms = set(window["logits"])
        if reference not in arms or (expected_arms is not None and arms != expected_arms):
            raise ValueError("every window must have the reference and identical arms")
        expected_arms = arms
        files = {}
        try:
            for arm, filename in window["logits"].items():
                source = path.parent / filename
                if source.stat().st_size != count * vocab * 2:
                    raise ValueError(f"wrong dump size: {source}")
                hashes[str(source)] = file_sha256(source)
                files[arm] = source.open("rb")
            rows = {arm: [] for arm in files}
            for i in range(count):
                logits = {arm: read_row(source, vocab) for arm, source in files.items()}
                next_index = start + i + 1
                target = tokens[next_index] if next_index < len(tokens) else None
                for arm in files:
                    rows[arm].append(row_scores(logits[reference], logits[arm], target))
            windows.append({
                "id": window["id"], "start": start, "rows": count,
                "nll_rows": sum("nll" in row for row in rows[reference]),
                "arms": {arm: aggregate(values) for arm, values in rows.items()},
            })
            for arm, values in rows.items():
                pooled.setdefault(arm, []).extend(values)
        finally:
            for source in files.values():
                source.close()
    result = {
        "scope": "same-history teacher-forced distributions; not free-running text agreement or task score",
        "manifest": manifest, "dump_sha256": hashes,
        "windows": windows,
        "pooled_by_position": {arm: aggregate(rows) for arm, rows in pooled.items()},
        "bootstrap_unit": "paired window means; descriptive only, no hard adoption threshold",
    }
    if "nvfp4" in pooled and "hq" in pooled:
        result["paired_NV4_minus_HQ"] = paired_windows(windows, "nvfp4", "hq")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", required=True, type=Path, help="new JSON file; existing files are refused")
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    result = run(args.manifest)
    with args.out.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(args.out)


if __name__ == "__main__":
    main()
