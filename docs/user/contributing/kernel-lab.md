---
title: "Kernel Lab"
description: "Iterate on kernels with a resident model, local replay and full-model evaluation."
---

Use Kernel Lab when one agent is repeatedly editing Trellis kernels against a
fixed model and benchmark. The model stays on the remote GPU between commands.
Run commands from the Laptop worktree; they handle source sync and remote
execution through `scripts/remote.py`. Only one agent can use the session and GPU
at a time.

Hot loading covers Trellis linear/SwiGLU operations: GDN/GQA projections, MLP
and the output head. Attention, GDN state updates and normalization remain fixed.

## Four commands

For first-time setup, create `.scratch/lab-settings.json` using the
[configuration example](kernel-lab-reference.md#configure-once).

| Command | Purpose |
| --- | --- |
| `python3 scripts/lab.py setup --config .scratch/lab-settings.json` | Establish the baseline, load the model and record or reuse operator inputs from one fixed benchmark request. |
| `python3 scripts/lab.py replay` | Sync, compile and hot-load kernel edits. Check each recorded input once, then time fixed representative calls. |
| `python3 scripts/lab.py forward` | Run the selected request through full prefill and generation; reuse a passing replay for the same candidate. Add `--full` to evaluate all requests with fresh baseline controls. |
| `python3 scripts/lab.py stop` | Stop the session and release the GPU; keep recordings, results and build caches. Also works during a busy evaluation. |

## Typical workflow

Run `setup` once, then repeat **edit → replay** for quick local feedback.
Use `forward` by default for full-model feedback on the request selected by
`benchmark` at setup. Reserve `forward --full` for acceptance checks that require
all benchmark requests and fresh baseline comparisons.
Continue editing or run `stop` when finished. Both evaluation commands restore
the same baseline afterward, even when the candidate passes.

Hot edits cover the linear, decode, prefill and T=8 MLP CUDA files and their
[supported headers](kernel-lab-reference.md#source-baseline-and-recovery). Other
code or configuration changes require stopping and [creating a new baseline](kernel-lab-reference.md#source-baseline-and-recovery).

Inspect the full result linked by each command under `.scratch/lab/`.
Report correctness, measured coverage and timings. Replay measures isolated
operators; forward measures the complete model, excluding HTTP service overhead.

For capture scope, storage budgets, failure recovery and measurement details,
consult the [reference](kernel-lab-reference.md). The
[design](../../specs/continual-experiments/README.md) explains implementation decisions.
