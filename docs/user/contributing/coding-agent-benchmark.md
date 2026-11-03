---
title: "Coding-agent decode baseline"
description: "Replay three frozen real SWE-bench requests at C1, and check 131,072-token input capacity separately."
sidebar:
  order: 8
---

## Run the fixed requests

`ignis-bench coding-agent` replays three independent SWE-agent request snapshots
from SWE-bench Verified. It measures the next response only: no tools execute,
no generated history passes to another request, and no SWE-bench score is computed.
The [fixture manifest and extraction notes](../../../crates/bench/fixtures/coding_agent/README.md)
record the sources, revisions, turns and hashes.

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
committed output token, including reasoning, text, tool syntax and EOS. The
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
