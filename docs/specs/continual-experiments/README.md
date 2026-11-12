# Resident experiments for the fixed 27B model

## Problem and source

A GPU job currently freezes/transfers source, waits for the GPU lease,
configures/builds the native leaf and Rust binary, starts a new CUDA context,
loads weights, allocates state, captures graphs, warms up, then measures.
Queue time, compilation, readiness, warmup and measurement must be reported
separately. A total job duration is not an engine-startup measurement.

[Together Compiler](https://www.together.ai/blog/together-compiler-continual-compilation-for-ai-inference)
(2026-11-04) retains engine state while replacing candidate computation and
uses progressively more representative measurements. Those two ideas fit this
fork. Its fleet-wide parallelism search is outside our single-GPU problem.

## Experiment purpose and lifecycle

An experiment answers a bounded performance question, for example: “Does a
T=8 MLP scheduling change reduce full decode latency without changing the
numerical result?” Record the question, baseline/source identity, allowed
change scope, workload and metric, numerical acceptance rule, time/candidate
budget, and stopping rule in `.scratch/` before starting. A question can admit
many candidate implementations. Close it with a supported, rejected or
inconclusive conclusion and the evidence; a fast layer alone does not settle
a full-decode question.

The [glossary](../../../GLOSSARY.md) distinguishes four units:

| Unit | Lifetime and ownership |
|---|---|
| Experiment | The research question, owned by the agent coordinating its conclusion. It can span several jobs, sessions and final service checks. |
| GPU job | A frozen queue submission. It owns the execution timeout, source isolation, GPU lease and result directory. The existing `remote.py experiment` command creates this unit. |
| Resident session | One loaded engine process within a job. Its engine binary, artifact and load settings stay fixed. A job can start multiple sessions sequentially, each with its own setup. |
| Candidate | One immutable source/library variant. Loading it creates a new engine generation on success; generation is a session-local counter, not the candidate's durable identity. |

The intended lifecycle is:

1. Define the question and controls; prepare the engine and initial candidates.
2. Acquire the GPU lease, check the card, load the fixed model and create a
   session. Establish the baseline for its measurement cases.
3. Iterate: edit a development worktree, freeze the candidate, compile a new
   library, check its contract, hot-load it, warm up and run the required
   measurement levels, save the result, and restore the baseline. Use that
   result to choose the next candidate. The model stays loaded throughout
   compatible iterations.
4. End the session when the candidate batch finishes, the budget expires,
   cancellation is requested, the engine must change, or a GPU failure requires
   recovery. Destroy the engine and release the lease. Do not leave an idle
   engine holding the card indefinitely while agents think or edit.
5. Validate promising candidates with the full GPU profile and rebuilt service
   benchmark, then record the experiment's conclusion. Full service checks
   need their own exclusive card use; stop the lab before running them.

Step 3's feedback-driven delivery of newly edited candidates is the target
workflow. The current queue and batch driver implement only a predeclared
candidate batch; the availability table below is normative about that limit.
These are workflow terms, not new CLI commands or request-schema fields.

“Setup once” means once per resident session, not necessarily once per research
experiment. Candidate compilation, library hash/ABI/workspace checks, graph
recapture, warmup and numerical/performance measurements still happen for each
candidate. A configuration change or recovery starts a new session and pays
setup again. Full workspace checks and the full GPU profile are acceptance
work, not mandatory repetitions in every compatible candidate's inner loop;
select the checks required by the change and keep final acceptance intact.

## Current capabilities and target work mode

| Capability | Current implementation |
|---|---|
| Agents edit and submit from independent worktrees | Supported; the queue freezes each submitted source and executes GPU jobs serially. |
| One session tests multiple immutable libraries | Supported; `scripts/kernel_lab.py` accepts repeated `--candidate` arguments and one fixed measurement plan. |
| Engine accepts serial reload commands while alive | Supported by the example's JSONL stdin primitive until EOF. It needs one controlling process. |
| Read feedback, edit, then deliver a new candidate to that running queued job | Not implemented by the queue/driver. The batch candidate list is fixed at invocation; no candidate inbox or remote stdin attachment exists. |
| Multiple agents submit candidates to one shared resident session | Not implemented. There is no multi-client session coordinator. |
| Reuse a model between separate queue jobs | Not implemented. The driver closes its engine; job completion requires descendant cleanup and GPU release. |
| Compile candidates in parallel without holding the GPU lease | Target behavior. Current queue jobs hold the lease across all their steps, including builds. |

The target mode retains independent worktrees and parallel candidate
preparation. A single controller owns a bounded resident session and serializes
candidate admission, reload, warmup, measurement and baseline restoration as
one unit. Each result must identify the originating source snapshot, candidate
binary, engine, artifact, settings and measurement plan. A failed or uncertain
attempt remains visible; recovery starts a new session and re-establishes its
baseline before further comparisons.

Until that controller exists, prepare a finite candidate batch in one frozen
job, inspect its results, then submit a new job for the next adaptive batch.
Do not edit the running job's tracked source: `scripts/experiments/runner.py`
checks it against the submitted snapshot and rejects such changes. Laptop
worktree edits do not update that snapshot. The future delivery path must
freeze new candidates separately while leaving the resident engine's source
and configuration fixed; it must not bypass queue isolation or GPU ownership.

## Boundary

Keep the model, weights, stream, KV/GDN pool, scheduler and memory ownership in
Ignis. Replace the two Trellis entry points used by `trellis_body`: linear and
SwiGLU. A library contains just the three existing Trellis translation units
and a versioned function table. This covers the current decode MLP search and
other Trellis projections without exposing the model's C++ layout to plugins.

Replacing the whole Rust engine would require a new state-transfer and ABI
contract. Replacing the whole C++ leaf would also couple old model handles to
new internal layouts. Loading individual cubins would require a second launch
path for the existing CUDA-runtime wrappers, including PDL. The two existing
Trellis functions already take device views and caller-owned workspace, so
this boundary needs no model-state migration.

The experiment engine is a Rust example using the normal model/step ABI. It
has a serial JSONL control channel, no HTTP route and no production deployment
action. A replacement is model-local. The default service uses built-in code.
Changing Rust scheduling, attention, KV format, quantization, model topology,
weights, or the workspace budget still requires a cold engine. This is an
explicitly bounded implementation, not arbitrary forward-pass replacement.

The replacement scope is narrower than the experiment's research scope:

| Change | Session action |
|---|---|
| Trellis linear/SwiGLU kernel implementation, preserving the public ABI, weight layout and numerical contract, fitting the existing workspace | Build a new immutable library and hot-load it at an empty-pool boundary. Recapture all decode/verify graphs and validate outputs. |
| Prompt, residual or output budget within the loaded envelope | Define a separate measurement case/reference. Keep each baseline/candidate comparison on the same case. The batch driver's plan is fixed for its run. |
| Model/weights, KV format, configured context/chunk/lanes, speculation or workspace capacity | Start a new session with explicit settings; hot loading cannot change these reservations. |
| Rust engine/scheduler, attention or another operator outside the plugin, or an incompatible ABI/layout | Rebuild the engine and start a new session. The candidate builder only compiles its three Trellis units; other source edits do not reach the resident engine. |
| CUDA execution error or uncertain device state | End the session and recover in a fresh process. Preserve the failed attempt and recheck the baseline. |

Reload requires an empty sequence pool on its owning model thread. It checks
the ABI and every actual model shape/width against the reserved workspace,
synchronizes the stream, loads and prepares the candidate, and captures new
decode and verify graphs. Old graphs and their library stay alive until the
replacement succeeds. Missing symbols, ABI/workspace refusals or incomplete
capture retain the old generation. Resetting all workspace bytes prevents
candidate scratch from contaminating the next arm. CUDA execution failures
can poison the context; terminate that session and start a fresh one.

A native plugin is trusted executable code. ABI validation catches accidental
incompatibilities; it is not a sandbox or proof of numerical correctness.
The caller never overwrites a loaded library. Content-addressed builds record
source/header hashes, compiler version, flags and library SHA-256. Source and
artifact identity remain the remote experiment queue's responsibility.

## Measurement levels

| Level | Measurement | Purpose and limit |
|---|---|---|
| 0: contract and operator | CPU request/ABI checks, candidate compilation, existing Trellis numerical GPU tests, workspace admission | Reject malformed and numerically wrong kernels before expensive runs. Use BF16/f64 oracles; random/synthetic inputs alone do not establish model behavior. |
| 1: layer | Resident GDN/GQA layer with supplied BF16 residual, fresh sequence state, widths 1/8/16 and prefill widths | Cheap integration feedback for projections, activation layout and residuals. Record input hash/provenance. A position-zero replay does not reproduce a long-context recurrent state or cache pressure. |
| 2: forward | Resident full 64-layer prefill/decode, actual token arrays, speculation off and DFlash2, one/two lanes | Detect graph, state, speculative and full-model regressions. Record logits hashes, output tokens, per-round widths/times and acceptance effects. Fixed-output-budget timing is diagnostic, not the service benchmark. |
| 3: service | Rebuilt production binary, frozen coding-agent requests and settings, canary/capacity and lifecycle checks | Confirm actual latency and correctness before integration. Include Vision/tools/cancellation/reload when affected. Only this level can support a production performance claim. |

The layer and forward levels share one resident full-model load. This fixed
artifact fits the card, so a separate few-layer loader would add ownership and
binding complexity without improving per-candidate startup. The layer command
executes only its selected layer; it does not reduce the initial weight load.

The supplied driver runs level 1 before level 2, stops a candidate at the first
numerical mismatch and restores built-in code. One warmup and three measured
samples are defaults, configurable in the experiment CLI. Built-in controls
bracket each candidate. Raw samples and drift ratios are retained; the driver
does not automatically choose or deploy a winner.

The initial gate requires exact layer output, prefill-logit hashes and decoded
tokens. This intentionally rejects some numerically valid arithmetic changes.
Such changes need a named numerical oracle/tolerance and the existing full GPU
profile, not a switch that disables checks. An exact token match by itself is
insufficient because it can hide large logit changes.

## What fixing the model removes

The artifact fixes tensor shapes, K distributions, layer types (48 GDN and 16
GQA), hidden width, vocabulary and weight layout. The device fixes SM120a.
These become the compatibility envelope and replay corpus, eliminating model
architecture discovery, graph partition search, distributed parallelism and
multi-platform dispatch from this implementation.

Traffic remains variable: context length, prefill chunk tails, lane count,
DFlash2 acceptance, prompt reuse and Vision change the work. The fixed model
therefore reduces the search space; it does not justify benchmarking only one
synthetic T=8 matrix or treating a layer speedup as an end-to-end speedup.

## Resource and promotion policy

One 5080 means one active engine. Run a finite candidate batch inside one frozen
`remote.py experiment` job and its inherited GPU lease; do not run a persistent
engine beside serving or escape the queue's cgroup. A session can test several
libraries while amortizing the load. Separate queue jobs still start cold.
Reusing a resident engine across separately submitted queue jobs is deferred:
it would change source isolation, failure recovery and lease ownership.

Use the existing fixed coding-agent baseline as the level-3 denominator.
Measure experiment-throughput gains separately from inference gains. This
change adds infrastructure and makes no model-throughput improvement claim.

## Acceptance

- Workspace CPU tests, Python tests and CUDA all-targets compilation pass.
- On a single idle RTX 5080, one process loads the real artifact and runs
  built-in → candidate → built-in with identical outputs and unchanged model
  and pool allocations, with speculation off and on.
- ABI, checksum, workspace, active-sequence and capture failures do not advance
  generation or corrupt the old engine.
- A changed computation is observed after replacement; stale CUDA graphs cannot
  silently make the same-code parity test pass.
- Startup and each replacement are timed separately, and all failed attempts
  remain in the evidence record.

See the [usage guide](../../user/contributing/kernel-lab.md) for commands.
