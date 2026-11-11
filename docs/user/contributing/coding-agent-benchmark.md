---
title: "Coding-agent decode baseline"
description: "Replay three frozen real SWE-bench requests at C1, and check 131,072-token input capacity separately."
sidebar:
  order: 8
---

## Current optimization baseline

On 2026-11-04, the project owner designated this run as the starting baseline
for RTX 5080 coding-agent decode optimization. Baseline ID:
`rtx5080-coding-agent-v1-2026-11-03`; measured code:
`e25780e21b240bca965b17bc6dbd600f170744e3`.

The reference medians are Astropy **207.657749750205 tok/s**, Django
**184.81640248687216 tok/s**, and SymPy **189.94180061893974 tok/s**.
For each task, report speedup as the new three-repeat median divided by its
reference median, and percentage change as `(speedup - 1) * 100`. Keep input
and output token counts, finish reasons, prefill time and speculative counters
beside the rates so a changed response length or acceptance pattern is visible.
Do not silently replace this baseline when a later run is faster.

The [archived measurement](../../findings/2026-11-03-coding-agent-5080-baseline.md) provides a repeatable baseline for later inference
changes: keep the fixtures, model, cache policy and generation protocol fixed,
and compare each task's full-window rate. The capacity pass remains a separate
check. The first report's short, synthetic requests and SSE timing define a
different workload and cannot be used as a direct speedup denominator.
Keep the hardware and serving configuration fixed too; if an experiment changes
one of these controls, identify it explicitly and report it as a separate
comparison. Retain the existing sanity check and the independent exact 131,072
input-token capacity check alongside subsequent performance results.

The [T=8 MLP scheduling finding](../../findings/2026-11-11-rtx5080-t8-mlp-scheduling.md)
records the output-partition and PDL optimization, its measured scope and
separate incremental comparisons. It does not replace the reference medians.

## Run the fixed requests

`ignis-bench coding-agent` replays three independent SWE-agent request snapshots
from SWE-bench Verified. It measures the next response only: no tools execute,
no generated history passes to another request, and no SWE-bench score is computed.
The [fixture manifest and extraction notes](../../../crates/bench/fixtures/coding_agent/README.md)
record the sources, revisions, turns and hashes.
The [first RTX 5080 result](../../findings/2026-11-03-coding-agent-5080-baseline.md)
records the measured configuration, per-request rates and separate capacity pass.

Build the server and bench on the GPU machine, using the
[remote entry point](remote-dev.md) from the Laptop:

```sh
python3 scripts/remote.py sync
python3 scripts/remote.py exec -- make build CUDA=1 UI=0 PROFILE=release
python3 scripts/remote.py exec -- cargo build --locked --release -p ignis-bench --target x86_64-unknown-linux-gnu
python3 scripts/remote.py exec -- make gpu-status
python3 scripts/remote.py exec -- make gpu-guard
python3 scripts/remote.py exec -- make start CUDA=1 UI=0 PROFILE=release \
  BIND=127.0.0.1:18080 IDLE_UNLOAD=0 LOOP_GUARD=off REQUEST_TIMEOUT=1800 \
  ARGS='--reuse-prompt false --reuse-retained-device 0 --reuse-retained-host 0'
python3 scripts/remote.py exec -- make gpu-status
python3 scripts/remote.py exec -- target/x86_64-unknown-linux-gnu/release/ignis-bench \
  coding-agent --endpoint http://127.0.0.1:18080 --out .scratch/coding-agent/decode
```

Configure the validated artifact in remote `local.mk` first. The server keeps
the card's profile (including speculation and decode lanes); **C1 means one
in-flight request**, not a one-lane load. GPU status before the bench must show
only this daemon. Do not run another workload or client during measurement.

The protocol is fixed: one full warmup per request, then three sequential
measurements; natural prompt lengths; reasoning enabled with effort `max` and
no thinking budget; greedy temperature 0, seed 0, top-p 1, top-k 0 and zero
presence/frequency penalties; 8,192 output tokens maximum. EOS is honored and
no stop strings are added. The model's usual reasoning-EOS redirect still
applies. A length finish is recorded as truncated, never silently removed.

Prefix reuse is disabled so repetitions remain the same workload without
editing their text. Every sample must report zero cached tokens and computed
prefill equal to the separately tokenized input. The runner refuses enabled
reuse and loop guards. Config, loaded model shape and available speculative
counters are recorded. The client deadline is the bench's existing 1,800 s
default (`IGNIS_BENCH_REQUEST_TIMEOUT`); the server deadline is recorded too.

## Read the measurement

`run.json` contains each warmup, all nine samples and each request's median.
The directory also keeps the exact sent requests and complete raw responses,
including reasoning and tool arguments. Failures, empty outputs, truncation
and inconsistent counts remain visible. An invalid sample prevents its
three-sample median and makes the command exit nonzero. A timed length finish
still contributes, labelled `truncated: true`. Existing output directories are
refused. Partial runs are saved after every response with `passed: false`.

The main rate is `usage.ignis_timing.output_tokens * 1000 / decode_ms`.
The clock starts **inside the scheduler at the final prefill chunk**, before
admission onto the decode lane and the first decode round, and stops on the
model thread at scheduler completion. It includes all draft/verify rounds,
sampling, KV updates and intervening scheduling. The numerator counts every
output token, including reasoning, text and tool syntax, matching
`usage.completion_tokens`. Runtime strips the terminal EOS from that count;
the complete round that encounters EOS remains in the time window. The
first anchor is sampled in prefill and committed in the first timed decode
round; it is included, and there is no `N-1` correction.

Prefill time is model-thread submission to that boundary, including admission
wait (C1 has no competing request). HTTP total time additionally includes
tokenization, transport and response decoding. These are separate fields.
The last prefill and first decode can occur within one scheduler call, so
neither first SSE arrival nor SSE chunk counts define these metrics. Tool-call
buffering cannot shorten this window. The old `small-card` client-side rate
uses a different boundary and is not directly comparable.

## Check 128K capacity separately

```sh
python3 scripts/remote.py exec -- make gpu-status
python3 scripts/remote.py exec -- target/x86_64-unknown-linux-gnu/release/ignis-bench \
  coding-agent --capacity-only --endpoint http://127.0.0.1:18080 \
  --out .scratch/coding-agent/capacity
python3 scripts/remote.py exec -- make gpu-status
python3 scripts/remote.py exec -- target/x86_64-unknown-linux-gnu/release/ignis-bench \
  canary --endpoint http://127.0.0.1:18080 --out .scratch/coding-agent/canary.json
python3 scripts/remote.py exec -- make stop
```

The capacity request reuses the existing TTFT prompt generator and server
tokenizer to create **131,072 tokens after chat-template expansion**. Reasoning
is off, the output cap is 64, and EOS remains enabled. It must complete the
whole prefill and generate nonempty output, with both tokenize and server
usage confirming the exact count. Its record and synthetic request are separate
from the three real requests' speed results. This checks capacity, not long
context answer quality. Canary is the existing sanity check.
