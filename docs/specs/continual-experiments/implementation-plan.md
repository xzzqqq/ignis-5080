# Implementation plan: resident exploratory kernel iteration

Current design: [README](README.md). User contract:
[guide](../../user/contributing/kernel-lab.md) and
[reference](../../user/contributing/kernel-lab-reference.md).
This revision supersedes the original four-command, fixed-corpus gate plan.

## Components

- The Laptop CLI owns receipts, command lock, preflight and existing remote sync.
- The pinned remote owner owns the GPU lease, resident engine, immutable source
  snapshots, candidate identities, scope cache and result persistence.
- The existing Rust JSONL/CUDA engine owns capture, replay, model forward,
  numerical diagnostics and checked CUDA-profiler calls. Strict native checks
  remain the default; exploratory requests opt into reporting output errors
  while still collecting timing.
- Nsight Systems injects into that same process at launch only after explicit
  `setup --enable-profiling` and
  collects only a requested replay range. It is optional for ordinary lab work.

No second engine, scheduler, resumable task database, new compiler cache or
parallel GPU execution is introduced. Libraries are trusted native code;
validation catches accidental incompatibility, not malicious code.

## Implementation order

1. Make setup model/configuration/request readiness independent of a permanent
   capture scope. Resolve repeatable operator and tokens-per-call selectors per
   operation; lazily publish compatible complete shards with budgets and hashes.
2. Make exploratory replay retain errors and timing for changed numerical
   outputs. Preserve strict native regression behavior. Keep representative
   selection candidate-independent and mutable restoration outside event timing.
3. Persist immutable named candidates. Support listing and fresh comparison of
   saved, current and builtin implementations on the same selected records.
   Quick forward reports the candidate's own trajectory and timing without
   promoting exploratory replay into an acceptance gate.
4. Default to a direct engine launch. Opt into a profile-capable resident session
   with `setup --enable-profiling`, checking Nsight before heavy setup. Keep the
   mode in session/results rather than baseline/corpus identity; stopped toggles
   reuse existing caches. Warm up
   outside collection; collect replay under CUDA profiler start/stop, stop and
   export, then parse kernel intervals. Retain raw report, SQLite, replay samples
   and machine timeline. Missing tools or empty traces fail explicitly.
5. Add `profile --forward` using original candidate-generated requests without
   a corpus or mandatory builtin control. Recapture candidate graphs once within
   the bounded capture under setup annotations, then collect actual samples.
   Join explicit runtime/NVTX and graph-node source/clone provenance; report
   phase/round/layer/operator instances, separate setup/unplaced work and
   coverage. Repeated layers aggregate per occurrence, never across round gaps.
6. Update all public command help and user docs around quick exploration,
   dynamic selection, candidate comparisons, profiler overhead and correctness
   boundaries. Remove the separate proposed workflow document.

## Validation obligations

CPU tests must cover deterministic scope identity/defaults, shard reuse and
rejection of corrupt/partial data, immutable candidate lookup, source/busy checks,
recovery, default direct launch even with Nsight installed, explicit enabled
launch, stopped mode toggles preserving identities, rejected irrelevant flags,
and profiler prerequisites, cleanup and interval arithmetic. Forward attribution
fixtures cover eager scopes, registered NVTX text, graph clone ancestry, repeated
rounds/chunks/samples, process/context correlation collisions, missing provenance
and exclusion of setup recapture from request metrics. Mocked
profiler tests cannot establish GPU collection or timing.

Before any GPU launch, use `make gpu-status` through `scripts/remote.py`. On an
exclusive RTX 5080, validate setup → scoped replay → changed candidate replay →
save/list/compare → quick forward → profile → second scope → stop, preserving
the same engine PID/model and builtin baseline across ordinary commands.
Check that lazy recording is excluded from measured/profile ranges and repeated
scopes reuse the cache. Demonstrate replay error reporting with an altered
candidate while preserving execution-error rejection and baseline recovery.

Inspect the raw Nsight report and exported kernel rows, confirm the intended
candidate launches, and compare profiler-off/profile-capable idle injection
costs before interpreting latency results. Instrumented timings are diagnostic;
ordinary replay comparisons remain the latency evidence. Nsight Compute
memory/occupancy collection is deferred.

Run the repository's required workspace CPU, Python, CUDA compilation and GPU
checks according to `docs/agents/testing.md`. Retain failures, unavailable checks
and actual measurements in the work record; no historical reload-only result or
synthetic trace substitutes for this revision's GPU evidence. HTTP/service
acceptance remains a separate workflow.
