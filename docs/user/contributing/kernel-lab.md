---
title: "Kernel Lab"
description: "Explore kernel edits with a resident model, real operator inputs and full-model feedback."
---

Use Kernel Lab when one agent repeatedly edits Trellis kernels against a fixed
model and benchmark request. Run commands from the Laptop worktree; they handle
sync and remote execution through `scripts/remote.py`. The model stays on the
remote GPU. One agent owns the session and GPU at a time.

Hot loading covers Trellis linear/SwiGLU operations: GDN/GQA projections, MLP
and the output head. Attention, GDN state updates and normalization remain fixed.

Create `.scratch/lab-settings.json` from the
[configuration example](kernel-lab-reference.md#configure-once), then run:

```sh
python3 scripts/lab.py setup --config .scratch/lab-settings.json
python3 scripts/lab.py replay --operator layer.0.down --tokens-per-call 8
```

Setup fixes the model, configuration and request. Select operators and token
counts per command. The first use of a scope records its real inputs; later
commands reuse compatible cached recordings. Recording costs are separate from
operator timing.

Repeat **edit → replay** for quick exploration. Replay reports timing and output
errors against the recorded baseline. Numerical differences do not block timing;
this is exploratory feedback, not strict correctness acceptance. Report the error
measurements along with speedups. Acceptance still needs the independent
correctness checks required by the repository.

Save useful candidates and compare them on the same selected inputs:

```sh
python3 scripts/lab.py replay --operator layer.0.down --tokens-per-call 8 --save-as before
# Edit the supported CUDA sources.
python3 scripts/lab.py replay --operator layer.0.down --tokens-per-call 8 --save-as after
python3 scripts/lab.py candidates
python3 scripts/lab.py compare --baseline before --candidate current --operator layer.0.down --tokens-per-call 8
```

`current` builds the current source; saved names select immutable candidate
libraries. `builtin` selects the session's pinned baseline. Inspect the result
linked by each command under `.scratch/lab/` for identities, coverage, raw samples
and errors.

Use `forward` for feedback on the selected full request. It reports the
candidate's own generation and engine times. Add `--full` for all requests,
configured repetitions and baseline controls. Replay timings describe isolated
operators and cannot establish full-model speedups.

Normal setup launches the engine directly with profiling disabled, even when
Nsight is installed. To inspect GPU kernel launches, restart with profiling
explicitly enabled:

```sh
python3 scripts/lab.py stop
python3 scripts/lab.py setup --enable-profiling
python3 scripts/lab.py profile --candidate current --operator layer.0.down --tokens-per-call 8
```

The profiling-enabled session uses Nsight Systems and retains the resident
engine across commands. Setup and evaluation results report `profiling_enabled`.
It reports kernel names, counts and durations, kernel span, gaps and
overlap, with a timeline and raw report. Profiling adds instrumentation overhead;
use replay/compare in a normal session for latency comparisons. Stop and run
`setup` without the flag to return to normal operation. Existing baselines,
recordings and named candidates are reused; no `--new-baseline` is needed. See the
[profiling details](kernel-lab-reference.md#operator-profiling) for prerequisites
and the meaning of each number.

Finish with `python3 scripts/lab.py stop`. It releases the GPU and keeps results,
saved candidates, recordings and build caches. Both evaluation and profiling
restore builtin code afterward. Unsupported source or configuration changes
require stopping and [creating a new baseline](kernel-lab-reference.md#source-baseline-and-recovery).

The [reference](kernel-lab-reference.md) covers configuration, coverage,
measurement and recovery. The [design](../../specs/continual-experiments/README.md)
explains the implementation boundaries.
