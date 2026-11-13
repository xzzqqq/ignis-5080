---
title: "Kernel Lab"
description: "Iterate on kernels with a resident model, local replay and full-model evaluation."
---

Use Kernel Lab when one agent is repeatedly editing Trellis kernels against a
fixed model and benchmark. The model stays on the remote GPU between commands.
Run commands from the Laptop worktree; they handle source sync and remote
execution through `scripts/remote.py`. One writer owns the session and GPU.

## Four commands

For first-time setup, create `.scratch/lab-settings.json` using the
[configuration example](kernel-lab-reference.md#configure-once).

| Command | Purpose |
| --- | --- |
| `python3 scripts/lab.py setup --config .scratch/lab-settings.json` | Fix the baseline, load the model and record or reuse real benchmark inputs for selected operators. |
| `python3 scripts/lab.py replay` | Sync and compile the current kernel edits, hot-load them, and check recorded operator calls for correctness and timing. |
| `python3 scripts/lab.py forward` | Pass the replay checks, then run the original benchmark prompts through full prefill and generation to compare correctness and model timings. |
| `python3 scripts/lab.py stop` | Stop the session and release the GPU; keep recordings, results and build caches. Also works during a busy evaluation. |

## Expected workflow

Run `setup` once, then repeat **edit → replay** for quick local feedback.
Run `forward` when a candidate is worth evaluating in the complete model.
Continue editing or run `stop` when finished. Both evaluation commands restore
the fixed baseline afterward; a passing candidate does not become the baseline.

Hot edits are limited to `kernel/src/trellis_linear*.cu`,
`trellis_linear_internal.cuh` and `trellis_decode.cuh`. Other code or configuration
changes require stopping and [creating a new baseline](kernel-lab-reference.md#source-baseline-and-recovery).

Inspect the full result linked by each command under `.scratch/lab/`.
Report correctness, measured coverage and timings: replay measures isolated
operators; forward evaluates the complete model inside the engine. Neither
measures HTTP service performance.

For capture scope, storage budgets, failure recovery and measurement details,
consult the [reference](kernel-lab-reference.md). The
[design](../../specs/continual-experiments/README.md) explains implementation decisions.
