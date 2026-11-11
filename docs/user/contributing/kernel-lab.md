---
title: "Resident kernel experiments"
description: "Compile and test Trellis candidates while the fixed model stays loaded."
---

## Scope

The `kernel_lab` example loads the 27B artifact once and accepts serial JSONL
commands. It can replace Trellis linear/SwiGLU code, recapture CUDA graphs and
restore built-in code. Weights and the sequence pool keep their allocations.
It does not add a service API or deploy candidates. Use the
[design](../../specs/continual-experiments/README.md) for the measurement levels
and compatibility limits.

CUDA builds and execution run on remote Linux. Start from an independent
worktree and use the [experiment queue](../../agents/gpu-experiments.md).
Place the following commands inside one experiment step so the engine keeps
the job's GPU lease. Run `make gpu-status` before GPU execution and verify
that only this experiment owns the card. The queue stops the whole campaign
on timeout, cancellation or worker failure.

## Build

```sh
cargo build --locked --release -p ignis-core --features cuda --example kernel_lab
python3 scripts/build_trellis_candidate.py --out .scratch/candidates
```

The builder compiles only `trellis_linear.cu`, `trellis_linear_decode.cu` and
`trellis_linear_prefill.cu`; it prints the immutable library path, identity,
cache-hit flag and elapsed seconds. Pass `--source /absolute/candidate/source`
to build another frozen candidate. The manifest beside the library records
source hashes, flags, compiler and binary checksum. Never modify a loaded `.so`.
Different source hashes produce different directories.

## Start and control

Create `settings.json`:

```json
{"max_context":4096,"prefill_chunk":128,"lanes":2,"kv_format":"bf16","speculation":"off"}
```

`speculation` accepts `off` or `dflash2` (window 7). Change settings by starting
a new session. Use `hq-e8-2b` for service-like KV and `bf16` for numerical
controls. This lab supports one or two lanes and at most 131,072 context tokens;
capacity remains subject to the GPU memory budget.

```sh
target/x86_64-unknown-linux-gnu/release/examples/kernel_lab \
  /path/to/qwen3_8_27b_trellis_sc30-v2.ninfer settings.json
```

The first output is a `ready` JSON object with startup phase timings. Send one
JSON object per line. EOF releases the engine. Example commands:

```json
{"op":"tokenize","text":"the exact rendered prompt"}
{"op":"measure","prompts":[[11,12,13]],"output_tokens":32}
{"op":"layer","layer":0,"tokens":1,"input":"/path/to/residual.bf16","output":"/path/to/result.bf16"}
{"op":"reload","path":"/absolute/candidates/DIGEST/candidate.so","sha256":"LIBRARY_SHA256_FROM_MANIFEST"}
{"op":"reload","path":"builtin"}
```

`measure` takes token arrays, not chat messages. Tokenize the exact rendered
prompt, including its chat template, before replay. The command starts fresh
sequences, records prefill-logit hashes, and generates a fixed number of tokens
without EOS stopping. It returns all tokens and per-round timings/widths.
Those rates are not comparable to the EOS-honoring coding-agent benchmark.

`layer` takes little-endian BF16 `[tokens][5120]` residuals with exactly that
byte count. Layer indices 3, 7, …, 63 are GQA; the rest are GDN. Every call
starts at position zero with fresh recurrent/KV state. Keep input provenance
beside the replay plan. A synthetic residual is useful for debugging but does
not establish behavior on real activation distributions or long-context state.
Layer timings include the synchronous layer ABI, excluding upload/download.

A successful reload advances `generation`; a refused library retains it.
Reload synchronizes, checks ABI/workspace compatibility and recaptures every
decode/verify graph. It requires no live sequences. Invalid numerical output
is detected by the comparison driver, which restores built-in code. A CUDA
execution error requires restarting the session.

## Ordered comparisons

Create a plan containing layer cases before forward cases:

```json
{
  "cases": [
    {"id":"gdn0","tier":"layer","command":{"op":"layer","layer":0,"tokens":1,"input":"/path/to/residual.bf16"}},
    {"id":"short-decode","tier":"forward","command":{"op":"measure","prompts":[[11,12,13]],"output_tokens":32}}
  ]
}
```

```sh
python3 scripts/kernel_lab.py \
  --engine target/x86_64-unknown-linux-gnu/release/examples/kernel_lab \
  --artifact /path/to/qwen3_8_27b_trellis_sc30-v2.ninfer \
  --settings settings.json --plan plan.json \
  --candidate /absolute/candidates/DIGEST/candidate.so \
  --out .scratch/lab-new
```

Repeat `--candidate` to test more libraries in the same process. Defaults are
one warmup and three measured samples per case; `--warmup`/`--samples` override
them. Output directories must be new. `run.json` preserves samples, exact
numerical verdicts, replacement time, bracketing baseline medians and drift
ratios. `engine.jsonl` and `engine.stderr` preserve the raw session. Nonzero
exit means a refused candidate, mismatch, or failed run, never a speed verdict.

A passing lab candidate still needs the normal Trellis numerical tests, full
GPU profile and the [coding-agent benchmark](coding-agent-benchmark.md) using
a rebuilt production binary. The lab's exact-output gate is deliberately
strict; arithmetic changes that legitimately alter bits need an independently
justified numerical comparison before promotion.

## Reload regression test

Build the real candidate above, then build six test-only fault libraries. Run
these commands inside the same exclusive remote experiment; do not load the
fault libraries in a service:

```sh
mkdir -p .scratch/faults
for mode in 1 2 3 4 5 6; do
  "$CUDA_HOME/bin/nvcc" -std=c++20 -shared --cudart=shared -Xcompiler=-fPIC \
    -DTEST_MODE="$mode" -Ikernel/include tests/fixtures/trellis_candidate_fault.cu \
    -o ".scratch/faults/fault-$mode.so"
done
make gpu-status
IGNIS_TRELLIS_ARTIFACT=/path/to/qwen3_8_27b_trellis_sc30-v2.ninfer \
IGNIS_TRELLIS_CANDIDATE=/absolute/candidates/DIGEST/candidate.so \
IGNIS_TRELLIS_FAULT_DIR="$PWD/.scratch/faults" \
  cargo test --locked --release -p ignis-core --features cuda \
    --test trellis_reload_gpu -- --ignored --nocapture
```

The test covers bad ABI, excess workspace, capture failure, prepare failure,
missing symbol, active-sequence refusal, repeated replacement and restore,
with speculation off and DFlash2 on. The fourth fault library deliberately
zeros results; the test requires zeros from later decode/verify rounds to
prove newly captured graphs execute the replacement code. It then restores
and checks the complete original observation again.
