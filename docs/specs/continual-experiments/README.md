# Single-writer resident kernel experiments

Design revision: 2026-11-17. Kernel Lab keeps one remote model resident while a
single Laptop writer explores Trellis kernel edits. The local wrapper uses the
existing `remote.py` sync/exec path, a pinned owner and private Unix socket, the
existing Rust/CUDA engine and candidate builder. It adds no inference engine,
compiler cache, task queue, public API or service benchmark tier.

The [user guide](../../user/contributing/kernel-lab.md) defines the commands and
[reference](../../user/contributing/kernel-lab-reference.md) defines their
measurement semantics. The [implementation plan](implementation-plan.md) records
the implementation boundaries and validation obligations.

## Session, scope and candidate

`setup` fixes model, runtime settings, prepared benchmark request and builtin
baseline. It prepares the resident engine and original request inputs.
Operator scope belongs to an operation: `--operator` and `--tokens-per-call`
select real calls without reloading the model. Omitted selectors mean all
supported operators/token shapes; legacy fixed-scope configuration is rejected. An uncached selection records its inputs under builtin code,
then destroys recording graphs and restores clean ones. Compatible complete
recordings are cached by baseline/model/request/settings/machine/scope identity.
Recording costs are reported separately from timing. Bounded pinned memory and
disk recording must fail clearly when a scope cannot fit, never silently sample.

`current` is an immutable library built from frozen synchronized source inputs.
`builtin` is the pinned baseline implementation. `replay --save-as NAME` retains
a candidate name and build identity; `candidates` lists names. `compare` resolves
both candidate selectors, checks all selected inputs once per immutable
candidate per invocation, and alternates representative timing legs on the
same scope with configured warmup for each leg. A saved name must not mutate the library behind another result.

One owner holds the host-wide GPU lease, supervises the resident engine and
builds, and serializes commands by rejecting busy/conflicting operations before
sync. Source identity is verified again remotely. Unsupported runtime/ABI edits
require stop and explicit `setup --new-baseline`. Ordinary commands restore
builtin functions and clean graphs before returning. Stop uses pinned control
and recorded process birth/target identity and can interrupt a busy operation.
CUDA failure, protocol failure, deadline or failed restoration ends the session.
An SSH disconnect does not end an accepted operation or trigger a retry.

## Exploratory feedback and correctness

Replay is quick timing plus numerical diagnostics, not strict acceptance.
Candidate output differences against frozen recorded outputs are retained, and
timing continues. Execution failure still fails the operation. Representative
inputs are selected deterministically from weight/view, shape, stride and
epilogue identity, independently of candidate performance. Mutable buffers are
restored outside timing. Native regression paths retain their strict checks;
exploratory behavior is explicit at the engine request boundary.

This distinction does not relax shipping correctness. Arithmetic changes still
need an independent reference and the repository's fixed acceptance contract.
Never use successful exploratory completion as a correctness verdict. Reports
retain scope, errors, raw timings, candidate identity and overhead components.
Local replay cache conditions differ from the model and timings cannot be summed
into a full-model speedup.

Quick `forward` runs the selected original request with candidate-generated
activations, state and tokens, and reports differences and engine timing. Full
mode retains all requests, repetitions and fresh baseline controls. Neither
mode injects captured intermediates or forces the builtin output trajectory.
HTTP/service overhead and service acceptance remain outside Kernel Lab.

## Resident operator profiling

Normal setup launches the engine directly, without Nsight or FIFO transport.
Only `setup --enable-profiling` launches the same engine under a named Nsight
Systems injection session. Missing `nsys` then fails before build/model loading.
Persist `profiling_enabled` in session state and every evaluation result, separate
from model configuration, baseline, corpus and candidate identities. Toggling
requires stop and setup; it preserves those caches without `--new-baseline`.
Starting after stop without the flag defaults to disabled. Bare setup on a live
session reports its mode; explicit enable on a live disabled session is refused. Injection and collection are distinct: no setup,
model loading or lazy-recording workload belongs in the requested profile.
The engine PID and model remain resident across profiles. A disabled session
rejects profile before compilation or recording, naming stop and
`setup --enable-profiling`. Injection overhead must be considered when comparing
latencies across sessions.

`profile` resolves the same candidate and selected real input representatives as
replay. Run unprofiled warmup, start a CUDA-profiler-API collection range, replay
selected samples, stop the range and export a `.nsys-rep` and SQLite timeline.
The engine exposes checked profiler start/stop calls. The collector stops even
when replay fails; normal host restoration still restores builtin code.

Machine summaries preserve kernel names, counts, durations, stream/context and
relative intervals. Per device, kernel span is the first start to last end,
busy time is interval union, gaps are span minus union, and overlap excess is
summed durations minus union. Summed duration is never wall time. The capture
includes replay setup/restoration kernels; CUDA event operator samples are
reported separately. Copies and host activity can occupy kernel gaps, whose
cause is not inferred. Empty/missing/unsupported exports are errors, not success.
Nsight Compute counters are outside this implementation.

## Evidence and implementation boundary

CPU tests cover scope/candidate orchestration, failure cleanup, protocol handling
and timeline interval arithmetic. GPU validation must cover two edits in one
resident process, real recording/replay, candidate use, rollback, saved/current
comparisons and bounded profiling artifacts. Report unavailable/failed checks;
mocked traces establish parser behavior only. New command latency and Nsight
injection overhead require measurement rather than inference from old reload
numbers. The existing [reload finding](../../findings/2026-11-12-resident-kernel-experiments.md)
is background evidence, not validation of this iteration's command latency.
