---
title: "Kernel Lab reference"
description: "Configuration, candidate comparisons, profiling and recovery."
---

See the [Kernel Lab guide](kernel-lab.md) for the edit/replay workflow.

## Configure once

Run from the Laptop worktree. CUDA builds and GPU execution use the configured
remote Linux checkout through `scripts/remote.py`. Use a dedicated checkout and
the private `.remote.json` / validated `.remote.env` described in
[remote development](remote-dev.md). The remote needs systemd user services and
user lingering. One transient systemd user unit owns the session's processes;
one RTX 5080 supports one session. Do not manually sync during a lab command.

Create an ignored local file, for example `.scratch/lab-settings.json`:

```json
{
  "artifact": "/path/to/qwen3_8_27b_trellis_sc30-v2.ninfer",
  "data_dir": "/large-disk/ignis-lab",
  "build_root": "/build-cache/ignis-lab",
  "build_jobs": 4,
  "benchmark": "astropy__astropy-12907",
  "settings": {
    "max_context": 65536,
    "prefill_chunk": 1024,
    "lanes": 2,
    "kv_format": "hq-e8-2b",
    "speculation": "dflash2"
  },
  "capture_pin_bytes": 1073741824,
  "capture_disk_bytes": 137438953472,
  "warmup": 1,
  "samples": 3,
  "timeout_s": 1800,
  "setup_timeout_s": 3600
}
```

Paths refer to the remote host. `data_dir` holds pinned baselines, source copies,
candidates, recordings and results. `build_root` holds the Cargo/CMake cache.
The context must fit each prepared prompt and its output budget. `benchmark`
fixes the request used by recording and quick forward. The default is
`astropy__astropy-12907`; alternatives are `django__django-10097` and
`sympy__sympy-11618`. `diagnostic_output_tokens` explicitly truncates generation
for functional checks; these runs cannot establish full-workload performance.

## Select inputs per command

Setup fixes model, runtime settings and request, without requiring one permanent
operator scope. Use repeatable `--operator` and `--tokens-per-call` selectors on
replay, compare and profile. Token count means rows in an individual operator
call, rather than the request's output budget. Selectors and actual coverage are
persisted with each result. Omitted selectors select all supported operators and
all token shapes; choose an explicit scope for quick feedback.

GDN layers expose `qkv`, `z`, `output`, `gate`, `up`, `down`. GQA layers
(3, 7, …, 63) expose `q`, `k`, `v`, `output`, `gate`, `up`, `down`.
`head` is the shared output head. Gate/up is one SwiGLU call.

The first command selecting an uncached scope records real calls from the fixed
request using builtin code. Compatible later commands reuse the recording.
Recording obeys `capture_pin_bytes` and `capture_disk_bytes` and available disk
space. It fails when the selected scope cannot fit; it does not silently sample
to fit. Narrow the scope or increase the budgets. Legacy `capture_names` and
`view` settings are rejected; replace them with command selectors. Changing scope
does not reload the model.

Recordings contain BF16 inputs, initial mutable targets and frozen outputs,
weight/view and shape/stride metadata, request and execution positions. Weights
are rebound in the resident model by stable identity; GPU pointers are never
persisted. Graph capture copies tensors at actual execution, then removes
recording nodes and rebuilds clean graphs. Incompatible, corrupted or partial
recordings are not reused.

## Replay and candidates

Replay is quick exploration: errors are reported, not strict acceptance gates.
It checks selected real inputs against their frozen outputs and times fixed
representatives of each distinct weight/view, token count, stride and epilogue.
It reports output error statistics even when numerical values differ and still
runs timing. Execution failures remain failures. Raw samples and numerical
results are retained. A completed exploratory replay does not approve numerical
changes for shipping.

Inputs stay on the GPU across timing repetitions. Mutable outputs and workspace
are restored outside CUDA event timing. Transfer, file I/O, recording and reload
costs are separate from operator samples. Representatives are selected by input
identity, never by a candidate's performance. Local cache conditions differ from
full forward; aggregate replay times are not full-model latency estimates.

`--candidate current` builds the synchronized source. `builtin` means the pinned
session baseline. `replay --save-as NAME` keeps the immutable compiled candidate
and its identity under a name. `candidates` lists saved candidates.
`compare --baseline NAME --candidate current` runs both candidates against the
same selected records and reports timing ratios and errors. Compare uses fresh
measurements; old setup timing is not its control. Keep small speedup claims
proportional to sample variability.

Compare alternates baseline/candidate order across pairs. `--samples` selects
the number of pairs; `--warmup` selects warmup executions for every leg.
Operator comparison checks all selected inputs once per immutable candidate
per invocation, then times fixed representatives in alternating legs.
`compare.checks` retains the complete diagnostics; every warmup/timing execution
also retains output diagnostics. No diagnostic result is cached across commands. Both flags also
override configured repetition counts on replay, forward and profile; warmup
may be zero and samples must be positive. Use
`compare --baseline before --candidate current --forward` for paired full-model
feedback on the fixed request, or add `--full` for all requests. Operator/token
selectors apply to operator comparisons and cannot be combined with `--forward`.

The builder caches CUDA objects by source, transitive headers, toolchain and
flags. T=8 MLP is a separate compilation unit, so editing it reuses generic
decode/prefill objects. Candidate libraries stay immutable. Results include
candidate identity, cache hits and compile/link costs.

## Full-model feedback

`forward` runs the fixed request from original prepared inputs, through natural
EOS or the original cap, using candidate-generated activations, state and tokens.
It reports engine prefill/decode timing and output differences. A quick run uses
one request and one sample. `forward --full` uses all three requests, configured
warmups/samples and fresh baseline controls. The fixtures retain tools and
history, normalization, artifact template/tokenizer, primary EOS, reasoning-close
behavior, greedy sampling and the 8,192-token cap; generated tools do not execute.

These are engine measurements, excluding HTTP and scheduler work. Do not compare
them directly with historical service TPS. Use the service workflow for service
acceptance and independent references for arithmetic changes.

## Operator profiling

Normal `setup` starts the engine directly, with profiling disabled even if
Nsight Systems is installed. Enable injection explicitly for a new session:

```sh
python3 scripts/lab.py stop
python3 scripts/lab.py setup --enable-profiling
python3 scripts/lab.py profile --candidate current --operator layer.0.down --tokens-per-call 8
```

The remote `nsys` executable must be in PATH when enabling profiling. If missing,
setup fails before building or loading the model. Disabled sessions reject
profile before compilation or recording, with the restart command. Missing tool
support, collection/export failure and empty kernel traces are failures.

The flag is a session option, separate from model configuration and baseline,
recording and candidate identities. Stop and run `setup` without the flag to
start a normal session; this preserves the pinned baseline and caches and does
not require `--new-baseline`. A bare `setup` on an existing live session reports
its current mode. Explicit `--enable-profiling` on a live disabled session asks
you to stop first. Setup/evaluation results report `profiling_enabled`.

A profile-capable engine starts under Nsight injection. Collection starts only
for the requested replay; setup/model loading and lazy recording are excluded.
The same resident weights, selected recorded inputs and candidate function table
are used. Warmup is outside collection. No second inference engine or model copy
is started. Injection can itself affect ordinary timing; compare like sessions
and report whether Nsight was active.

Results retain the `.nsys-rep`, SQLite export and machine-readable timeline.
The summary groups kernel names with counts and duration statistics. Per-device
kernel span runs from the first captured kernel start to the last kernel end;
busy time is the union of kernel intervals, gap time is span minus that union,
and overlap excess is summed kernel durations minus the union. Overlap excess
counts concurrency with multiplicity. Gaps can include copies, synchronization
and host work; they do not identify a cause. CUDA event replay samples provide
operator timings separately. Summing kernel durations is never operator wall
time. The captured envelope includes replay setup/restoration kernels as well
as the measured launches; inspect the timeline and counts when interpreting it.

Nsight Compute memory/occupancy counters are not collected by this command.
Profiler timings include instrumentation overhead and are diagnostic evidence;
use replay/compare for latency measurements.

## Source, baseline and recovery

Hot loading supports `trellis_linear.cu`, `trellis_linear_decode.cu`,
`trellis_linear_prefill.cu`, `trellis_linear_mlp_t8.cu`,
`trellis_linear_internal.cuh`, `trellis_decode.cuh` and
`trellis_decode_helpers.cuh` in `kernel/src/`. Other runtime/build/ABI changes
require a new baseline:

```sh
python3 scripts/lab.py stop
python3 scripts/lab.py setup --new-baseline
```

Repeated setup for a matching live session reports its state and preserves its
baseline. Recovery also preserves the pinned baseline. To change configuration,
stop first and provide the new `--config` with `--new-baseline`.

Ordinary commands restore builtin code and clean graphs. Numerical differences
are recorded. Sync/build/admission failures do not measure stale code. CUDA
failure, timeout, invalid engine replies or failed restoration end the session.
`timeout_s` bounds an accepted evaluation and `setup_timeout_s` bounds setup.
Lost replies are not retried; inspect saved results and logs.

A command returning leaves the engine resident. Concurrent evaluations are
refused. `stop` uses recorded ownership and pinned control code, without sync or
rebuild, and confirms the GPU process has disappeared. It can interrupt setup or
evaluation. Results, recordings and caches survive stopping.

Local receipts and returned results live under `.scratch/lab/`. Remote session
directories retain settings, inputs, engine logs, source/build identity and
operation results. Reports separate client preflight/sync/round-trip from owner
build/reload/recording/measurement. Keep the linked full results and do not delete
control receipts while the session runs.

## Development checks

CPU suites check orchestration and exported-timeline arithmetic. The opt-in real
GPU regression runs through the public commands:

```sh
IGNIS_LAB_E2E_CONFIG="$PWD/.scratch/lab-settings.json" \
  python3 -m unittest discover -s tests -p test_lab_gpu.py
```

The regression edits supported CUDA sources, restores them in `finally` and
stops its session. Run without another writer or GPU workload. A short diagnostic
configuration is suitable for functionality, not performance evidence. Before
launching any GPU check, run `make gpu-status` through the remote entry point.

Native `lab_capture_gpu` and `trellis_reload_gpu` ignored tests cover
recording/clean-graph parity, padded output, repeated ADD restoration, invalid
views, plugin use and rollback. Use the artifact and fault libraries documented
in their headers. See [testing](../../agents/testing.md) for required checks.
