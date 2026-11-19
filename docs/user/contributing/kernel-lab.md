---
title: "Kernel Lab"
description: "Explore kernel edits with a resident model, real operator inputs and full-model feedback."
---

Use Kernel Lab for repeated kernel experiments against a fixed model
and benchmark request. Run commands from the local worktree; they handle
sync and remote execution through `scripts/remote.py`. The model stays on the
remote GPU. Agents can collaborate, but one executor owns the session and GPU
at a time; coordinate access before issuing commands.

Hot loading covers Trellis linear/SwiGLU operations, the complete C1/T8 HQ
GQA attention launcher, and three finite layer segments (MLP, GQA and GDN). The same selected attention code runs in stateful replay
and real forward graphs. Other attention shapes use builtin code; GDN state
updates and normalization remain fixed.

NVFP4 settings are supported for full-model `forward --candidate builtin`
and `profile --forward --candidate builtin`. Stateful `layer.N.attention`
and `layer.N.gqa_block` recording/replay still require HQ's G32 cache and
reject NVFP4's G16 cache. This does not add an NVFP4 attention hot-load ABI;
the HQ attention/segment scope above remains unchanged. The configuration
example pins HQ so its replay examples use that supported scope.

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

For GQA attention, select one layer explicitly:

```sh
python3 scripts/lab.py replay --candidate current --operator layer.3.attention --tokens-per-call 8 --save-as gqa-before
# Edit kernel/src/lab_gqa_candidate_policy.cuh or supported GQA kernel sources.
python3 scripts/lab.py compare --baseline gqa-before --candidate current --operator layer.3.attention --tokens-per-call 8
```

This records the actual rotated Q/K, V, gate, device positions and mask, and
paged HQ KV plus residual/ring state before and after attention. Replay restores
an isolated copy before each measured launch and reports BF16 output errors and
exact state-byte differences separately. The initial scope is single-sequence,
eight-column HQ attention, including masked tails; a pool configured for two
lanes works. Recording all token widths or BF16 attention is unsupported. Full
physical state images can be large: at context 65,536 with two pool lanes, one
layer costs about 153 MiB per recorded call. Narrow the scope and budget disk
space for the complete fixed request. See the [reference](kernel-lab-reference.md#stateful-gqa-attention-replay)
for limits, candidate workspace ownership and supported source edits.

For a short diagnostic pass, explicitly limit the recording:

```sh
python3 scripts/lab.py replay --candidate current --operator layer.3.attention --tokens-per-call 8 --record-start 0 --record-limit 1
```

The limit is applied before device-to-host tensor copies and file writes. Results
label this coverage as a sample; a limited corpus never satisfies a later full
scope. To revisit a particular existing recording, copy its remote `shard` path
from the result and use `--corpus /remote/path/to/shard --record-start N
--record-limit 1` (matching records are ordered by execution ordinal). Both legs of `compare` use the same selected immutable record
IDs. New recordings include per-record hashes, so checking a sample reads only
its recorded byte ranges rather than hashing the entire data file.

Export a selected record, including the loaded candidate's actual output:

```sh
python3 scripts/lab.py export --candidate current --corpus /remote/path/to/shard --record-start 0 --record-limit 1
```

The result links a remote artifact containing inputs, frozen references,
candidate outputs, state and exact model weights, with shapes, dtypes, byte
ranges, hashes and candidate/source identities. Native candidates may also
provide internal diagnostic taps; their availability is explicit in the manifest.
See the [sampling and export reference](kernel-lab-reference.md#sample-records-and-export-tensors)
for ordering and integrity guarantees.

To explore a larger boundary, select a finite block:

```sh
python3 scripts/lab.py replay --candidate current --operator layer.0.mlp_block --tokens-per-call 8
python3 scripts/lab.py replay --candidate current --operator layer.3.gqa_block --operator layer.0.gdn_block --tokens-per-call 8
```

These boundaries include their input normalization and final residual add.
GQA/GDN blocks exclude the following MLP. GDN replay retains all four commit
record planes and verifies that persistent recurrent/conv state stays readonly.
Candidates own the complete launch sequence and temporary layout inside each
block. Block selectors cannot be mixed with legacy operator selectors in one
command; issue separate commands for those scopes. See the
[finite segment reference](kernel-lab-reference.md#finite-layer-segments).

Repeat **edit → replay** for quick exploration. Replay reports timing and output
errors against the recorded baseline. Numerical differences do not block timing;
this is exploratory feedback, not strict correctness acceptance. Report the error
measurements along with speedups. Run full acceptance on the accumulated changes
at the end of an optimization round, not after each replay candidate.

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

Use `forward` when full-request feedback helps diagnose issues or assess combined
changes; it is not required after every replay. It reports the
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
overlap, with a timeline and raw report. Add `--forward` to profile the fixed
request through the whole model:

```sh
python3 scripts/lab.py profile --forward --candidate current
```

Forward profiling reports prefill/decode phases, individual rounds, layers and
annotated operators. It uses the candidate's own activations and generation,
without recording an operator corpus. The default is one request sample and
no warmup. Coverage and unclassified work remain visible; inspect the linked
artifact for the complete breakdown. Profiling adds instrumentation overhead;
use replay/compare in a normal session for latency comparisons. Stop and run
`setup` without the flag to return to normal operation. Existing baselines,
recordings and named candidates are reused; no `--new-baseline` is needed. See the
[profiling details](kernel-lab-reference.md#operator-profiling) for prerequisites
and the meaning of each number.

Keep the session resident across experiments. When the session is no longer
needed or GPU ownership is handed off, use `python3 scripts/lab.py stop`. It
releases the GPU and keeps results, saved candidates, recordings and build
caches. Both evaluation and profiling
restore builtin code afterward. Unsupported source or configuration changes
require stopping and [creating a new baseline](kernel-lab-reference.md#source-baseline-and-recovery).

The [reference](kernel-lab-reference.md) covers configuration, coverage,
measurement and recovery. The [design](../../specs/continual-experiments/README.md)
explains the implementation boundaries.
