# Benchmark-derived Trellis recordings replay through a resident candidate

- Kind: experiment
- Status: current
- Observed: 2026-11-12
- Last verified: 2026-11-12
- Scope: kernel lab / RTX 5080, benchmark capture, persistent replay, process lifecycle
- Related: [resident design](../specs/continual-experiments/README.md), [implementation plan](../specs/continual-experiments/implementation-plan.md), [initial reload measurements](2026-11-12-resident-kernel-experiments.md)
- Superseded by: none

## Question

Can one writer repeatedly edit and test Trellis candidates against real
coding-agent inputs, with the model resident and intermediate data reusable
across sessions, without an experiment queue or a live terminal handle?

## Evidence

The implementation extends `fa40b3d8` in an independent worktree. Exact engine,
source, candidate and corpus identities are in the retained session/results.
Hardware is one RTX 5080 16 GB, driver 580.105.08, CUDA 13.0.88, Rust 1.99.1.
The artifact is the same validated Trellis container as the initial finding:
SHA-256 `2c4c9ee4d6b25c08da6c2d85840126e73d120564e230f926b6230b5b95ee8b3c`.

The full-generation run uses context 65,536, prefill chunk 1,024, two configured
lanes with C1 requests, hq-e8-2b KV and DFlash2 window 7. It honors primary EOS
and reasoning close, with the 8,192-token cap. The explicit capture scope is
`layer.0.gate/down/qkv`, `layer.3.q/k/output`, and `head`; this is selected
coverage, not a recording of every model projection. The replay view selects
T=8 calls. Warmup and samples are one each for this functional comparison.

| Request | Prepared input tokens | Generated tokens | Finish |
|---|---:|---:|---|
| Astropy | 19,954 | 913 | EOS |
| Django | 22,420 | 2,319 | EOS |
| SymPy | 16,717 | 1,379 | EOS |

Recording, clean baseline and the same-code candidate preserve the token
streams, prefill-logit hashes and speculative progress. Reasoning is closed at
the end of each request. The corpus contains 7,669 actual calls and
33,208,317,952 bytes of tensor data; 6,398 records match the T=8 view. Selected
prefill alone has a computed lower bound of 17,308,637,184 bytes. Three requests
therefore do not imply a small intermediate dataset.

The full candidate command's owner-side wall time is 310.49 s, including its
local gates and before/candidate/after full-forward controls. Its candidate
cache lookup/build takes 0.036 s and reload takes 18.19 ms. These measurements
are different scopes; reload time is not command latency. Candidate engine
median decode times are 4,524.90 / 11,809.61 / 5,913.77 ms for the three requests.
They are diagnostic engine-call timings, not the service benchmark's rates;
no inference improvement is claimed for identical code. After the final Python
control changes, another successful replay covers the same 6,398 records:
24.51 s client wall time (0.92 s preflight, 1.58 s sync, 22.02 s remote request).
Its local CUDA-event median sum is 895.69 ms; that sum excludes restoration,
transfers, graph creation, IPC and bracketing work and is not command latency.

The separate four-command end-to-end regression passes in 358.14 s with an
explicit eight-token diagnostic cap. It checks changed candidate identity,
compiler failure without losing the model, rejection of deliberately zeroed
SwiGLU output, recovery to the original code, full diagnostic forward, repeated
stop, and corpus reuse with a new engine PID. A lifecycle regression passes in
132.96 s: an accepted operation finishes once after client disconnection,
stop interrupts a busy evaluation, and killing the owner leaves no GPU child.
A later one-second evaluation deadline returns failed/finished with the deadline
error and no stale successful result (about 3.65 s including client preflight,
sync and remote cleanup in that observation).

Native GPU regressions cover all GDN/GQA projection roles, gate/up and down,
the output head, actual eager and verify-graph records, padded destination
strides, repeated ADD reset, invalid view rejection, and current-plugin use in
both local and complete forward. The canonical GPU profile passes 130 CTest
checks plus six tests in its five Rust suites with no skips, in 505.46 s.
The separate capture and reload GPU tests also pass.

Local workspace CPU tests pass 3,068 tests (six pre-existing ignores); Linux
passes 3,069 (six ignores). The final local Python suite passes 77 tests with
nine skips (seven Linux-only, two opt-in GPU tests); the two GPU integration
scenarios are separately executed above. Linux Python passes the same 77-test suite with only the two opt-in GPU
scenarios skipped there; those scenarios are executed separately above. Source changes after the full GPU
profile are Python control/cleanup and documentation only; relevant Python
checks and command-lifecycle tests cover them.

Raw evidence is local under `.scratch/lab-final-validation/`,
`.scratch/lab-e2e/commands.json`, `.scratch/lab-lifecycle/`,
`.scratch/lab-full-forward.log`, and the full operation record
`.scratch/lab/1380165bd0e246d7b5e08cfbe3e97545.json`. Corpus payloads and pinned
sessions remain in the configured remote `data_dir`; large tensors are not
committed or copied back to the Laptop.

## Finding

Observed: the existing Trellis replacement API can support both persistent
real-call replay and full model execution under four independent local
commands. A session outlives command/SSH clients; a single transient systemd
user unit supplies process-group cleanup without adding a job queue.

Observed: request preparation must share the serving normalization, not just
its tokenizer. The first preparer omitted tool defaults and produced inputs
12 tokens too short for every fixture. Reusing `resolve_tools` fixed the
counts above; the faulty development corpus is not used for the final run.
The preparer also uses the runtime's primary-EOS accessor and shared reasoning
budget state to avoid divergent termination rules.

Inference: explicit capture/measurement scope and storage budgets are essential
for this workflow. Persisting actual calls avoids repeated baseline capture,
but replaying every stored tensor can itself be expensive. Store coverage and
measurement coverage must remain distinct and visible.

## Implications

Use the [four-command guide](../user/contributing/kernel-lab.md). Keep source
identity, immutable candidate files, numerical gates and bracketing controls.
Use local replay for diagnosis and full forward for whole-model conclusions;
never force the candidate along recorded intermediate values in full forward.

## Limits and unknowns

- Full generation above used one measured sample per arm, selected capture
  scope and one hardware configuration. It is functional evidence, not a
  production performance gate or a claim of an optimized kernel.
- All-Trellis recording can require far more storage. The implementation
  refuses insufficient pin/disk budgets; automatic grouped recording is not
  implemented and no calls are silently sampled away.
- Only linear/SwiGLU computation is replaceable. Attention, convolution and
  recurrent-state history capture remain outside this boundary.
- Record `position` identifies the enclosing chunk/round; draft/head rows need
  their call shape and role to interpret it. This is a local replay corpus,
  not a general execution dependency trace.
- Exact comparison rejects some valid arithmetic changes. Such changes need
  an independent numerical oracle before acceptance rules can change.
- HTTP/service performance and capacity/canary acceptance remain outside this
  feature. The historical service baseline is unchanged.

## Follow-ups

The design and implementation plan linked above define the operating contract.
Further performance optimization should use captured inputs and report the
corresponding full-forward result, with capture scope and timing boundaries.
