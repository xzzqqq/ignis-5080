# Three frozen coding-agent requests on the RTX 5080

- Kind: experiment
- Status: current
- Observed: 2026-11-03
- Last verified: 2026-11-03
- Scope: serving / Qwen3.8-27B trellis on RTX 5080, C1 coding-agent decode and 128K input capacity
- Related: [first 5080 bench](2026-11-03-rtx5080-first-bench.md), [run guide](../user/contributing/coding-agent-benchmark.md), [fixture sources](../../crates/bench/fixtures/coding_agent/README.md), [ADR 0063](../adr/0063-the-profile-follows-the-card-class.md)
- Superseded by: none

## Question

What is the full decode rate of the next response to three fixed, real coding-agent
requests at C1 on this fork's RTX 5080? Can the same load prefill an input of exactly
131,072 template-expanded tokens and then generate?

## Evidence

Measured implementation commit: `e25780e21b240bca965b17bc6dbd600f170744e3`, with a
clean remote working tree. Decode ran on 2026-11-03 at 22:33:27–22:36:26
Asia/Shanghai, followed by capacity and canary through 22:37:49. Later changes
clarify the terminal-EOS counting convention in comments, a test and the guide;
they do not change the measured implementation's behavior. No inference algorithm
or kernel was changed for this benchmark.

### Requests and protocol

The source is the public SWE-bench Verified submission
`20250522_sweagent_claude-4-sonnet-20250514`, pinned to experiments repository
revision `40f164d5b8f1d249bf95a6df8b74b577fd8e519d`. The
[manifest](../../crates/bench/fixtures/coding_agent/manifest.json) records source
URLs and SHA-256 hashes, task revisions, extraction revisions and every fixture hash.

One task from each of Astropy, Django and SymPy was chosen before any Ignis run:
the first lexical task ID in each project, at a turn after reading relevant
implementation/tests and before modifying the target code. Selection used task
content, not rate or speculative acceptance. Input lengths were left natural.

| Task | Query index (zero-based) | Stage / bug | Fixture SHA-256 |
|---|---:|---|---|
| astropy__astropy-12907 | 16 | Nested CompoundModel separability; relevant source and tests read | `21a7d230baa23d348a737c71d93662d431d9f08aa52132cc08377427ab5ca7d4` |
| django__django-10097 | 13 | URLValidator credential regex; validator source and tests read | `6877d43ca8dd8edb9908be291fa4df8da68453e3a37c799435fcc008777b2ce9` |
| sympy__sympy-11618 | 15 | Point distances across dimensions; geometry implementation read | `0ac09494a44dac1bb8c50f636a967f245c7f37029f8d73789b300582adaea211` |

The extractor takes the selected trajectory entry's `query`, applying SWE-agent's
model-visible history projection. System, messages, prior tool calls and returns
are retained; the selected answer, its observation and future entries are excluded.
The archived query omits native tool schemas. These are reconstructed from public
SWE-agent v1.1.0 code/config and checked against every archived command name,
description and argument description. The trajectory's reported agent revision
`383965cc0e358f8e48b841d9b970ac96fe9cd335` is not publicly resolvable; the reconstruction
uses `0f3acafacabc0def8cc76b4e48acb4b6cf302cb9`. This is an explicit provenance limit,
not a claim of original provider wire-byte equivalence. S3 objects have no VersionId;
their downloaded bytes are pinned by SHA-256.

Each request had one complete warmup followed by three measured responses. All
requests were independent and sequential (C1); no generated tool ran and no output
was passed to another request. Fixed protocol: thinking enabled, reasoning effort
`max` (no request thinking budget), preserve thinking, greedy temperature 0, seed 0,
top-p 1, top-k 0, presence/frequency penalties 0, output cap 8,192, no stop strings,
EOS honored. The usual model reasoning-EOS redirect remains enabled. The server's
configured thinking budget is 32,768, overridden by request effort `max`.

Prefix reuse was disabled, retained device/host slots were zero, and every response
reported zero cached tokens, one full prefill pass and computed prefill equal to
the separately tokenized input. Warmups therefore do not warm a prompt cache.
Length, failure and empty-output cases are retained by the runner; none occurred
in these 12 responses. Both server and client deadlines were 1,800 seconds.

### Clock and token count

The old `small-card` rate uses `(completion_tokens - 1)` divided by time between
first and last SSE arrivals. That is unsuitable here: tool calls can be buffered,
and the final prefill and first speculative round can share one scheduler advance.

The new opt-in `usage.ignis_timing` uses a timestamp emitted inside the scheduler
immediately after the final prefill chunk, before decode admission and the first
round. It ends on the model thread at scheduler completion. Thus all actual
draft, verify, sampling, KV updates, intervening scheduling and completion costs
are included, independently of HTTP buffering. Prefill is timed separately from
model-thread submission to the boundary, including admission wait; C1 has no
competing request. HTTP total additionally includes tokenization, transport and
response decoding.

The rate is `output_tokens * 1000 / decode_ms`. The output count matches
`usage.completion_tokens` and includes reasoning, text and generated tool syntax.
The runtime strips terminal EOS from this count, but the entire round that
encounters EOS remains timed. The first anchor is sampled in prefill and committed
in the first timed decode round; it is counted, with no `N-1` subtraction. In these
EOS-ended runs, `spec.rounds + spec.accepted` is one larger than output tokens.
SSE chunks and decoded text lengths are never used as token counts.

### Machine, artifact and serving configuration

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5080, 16,303 MiB, 360 W limit, driver 580.105.08 |
| PCIe | Gen 5 x16 during the run |
| Host | Intel Core i9-13900K, Ubuntu 22.04 / Linux 5.19 |
| Build | Linux x86_64 release, CUDA, SM120a; Rust 1.99.1, CUDA 13.0.88, CMake 4.4.5, GCC 11.3.0 |
| Profile / served model | `rtx50-16gb` / `qwen3.8-27b-16g` |
| Context / lanes / prefill chunk | 262,144 / 2 configured lanes, C1 workload / 1,024 |
| KV / prefill attention / exception weights | `hq-e8-2b` / `fp8` / `bf16` |
| Speculation | DFlash2, 7 draft tokens, `auto` proposal head resolved to `full` |
| RoPE / loop guard / idle unload | `auto` resolved to `none` / off / off |
| Prefix reuse | off, retained device 0, retained host 0; configured host pool 8G |
| Vision | Profile remains enabled, max 4,096, host residency; all requests text |
| Other services | UI, speech and diarizer off |

The artifact is `models/qwen3_8_27b_trellis_sc30-v2.ninfer`, 13,379,877,124 bytes,
from `gpillon/Qwen3.8-27B-trellis-3bpw-ignis` revision
`f5bbd2cc14c9ca573ef5553c100ac69410f1fe58`:

- Artifact SHA-256: `2c4c9ee4d6b25c08da6c2d85840126e73d120564e230f926b6230b5b95ee8b3c`.
- Conversion sidecar: 309,495 bytes, SHA-256
  `210ab54bfe8b04e5fc4337ee3755c340ea3f21239aa625c505aa2406645a340d`.

The load adapted nothing and allowed no oversubscription. Its VRAM plan was
15,347,997,868 bytes against 15,348,203,520 bytes; KV had 4,851 pages / 310,464 tokens.
The 263 one-second GPU samples reported 14,413 MiB throughout, 46–84 °C, and peak
362.82 W. These span prefill and decode and are not decode-only utilization data.

### Decode results

Warmups are excluded from medians. Every measured finish reason was `tool_calls`;
every sample was valid, nonempty and untruncated, with cached tokens 0.

| Task | Input tokens | Output tokens (each repeat) | Median decode tok/s | Median prefill (s) | Median decode (s) | Median HTTP total (s) |
|---|---:|---:|---:|---:|---:|---:|
| astropy__astropy-12907 | 19,954 | 876 | **207.658** | 9.5444 | 4.2185 | 13.7877 |
| django__django-10097 | 22,420 | 954 | **184.816** | 10.9022 | 5.1619 | 16.0942 |
| sympy__sympy-11618 | 16,717 | 1,268 | **189.942** | 8.0841 | 6.6757 | 14.7838 |

| Task | Repeat | Prefill (s) | Decode (s) | HTTP total (s) | Decode tok/s |
|---|---:|---:|---:|---:|---:|
| Astropy | 1 | 9.5255 | 4.2058 | 13.7577 | 208.283 |
| Astropy | 2 | 9.5444 | 4.2185 | 13.7877 | 207.658 |
| Astropy | 3 | 9.5846 | 4.2197 | 13.8291 | 207.596 |
| Django | 1 | 10.8875 | 5.1619 | 16.0775 | 184.816 |
| Django | 2 | 10.9022 | 5.1617 | 16.0942 | 184.823 |
| Django | 3 | 10.9092 | 5.1632 | 16.1012 | 184.770 |
| SymPy | 1 | 8.0815 | 6.6744 | 14.7838 | 189.980 |
| SymPy | 2 | 8.0841 | 6.6757 | 14.7824 | 189.942 |
| SymPy | 3 | 8.0856 | 6.6767 | 14.7845 | 189.916 |

| Task | Spec rounds | Drafted | Accepted | Accepted / drafted |
|---|---:|---:|---:|---:|
| Astropy | 161 | 1,127 | 716 | 63.53% |
| Django | 191 | 1,337 | 764 | 57.14% |
| SymPy | 264 | 1,848 | 1,005 | 54.38% |

Spec counters were identical across each task's warmup and three repeats. Full
per-position arrays are in the raw JSON. The four outputs per task were identical
after excluding generated tool-call IDs. Astropy generated `str_replace_editor`;
Django and SymPy generated `bash`. All tool arguments parsed as JSON objects.
Reasoning and text were nonempty and retained. This inspection does not establish
that the proposed edits solve the tasks; no generated command was executed.

### Separate 128K capacity check and sanity

The synthetic capacity request contained **131,072 input tokens after template
expansion**, confirmed by both `/v1/tokenize` and response usage. Computed prefill
was also 131,072, cached tokens 0, prefill passes 1. It completed prefill in
**75.1966 s**, decoded one output token (`OK`) in 0.0372 s, and completed HTTP in
75.4507 s with finish reason `stop`. No OOM, truncation or empty output occurred.
Its output cap was 64 with thinking disabled and EOS honored. This is a capacity
pass and contributes nothing to the three decode medians.

Existing `ignis-bench canary`: **4/4 sane and greedy-deterministic**, self-consistency
PASS. As in the first 5080 report, two canaries reached an answer and two exhausted
their small budget while thinking. This is the existing sanity criterion, not a
new quality evaluation. The daemon was stopped afterward; the final GPU check
showed 0 MiB used and no compute process.

### Reproduction and archive

Configure the validated artifact through remote `local.mk`; `.remote.env` selects
the validated toolchain. All Laptop-to-GPU operations use the repository entry point:

```sh
python3 scripts/remote.py sync
python3 scripts/remote.py exec -- make build CUDA=1 UI=0 PROFILE=release
python3 scripts/remote.py exec -- cargo build --locked --release -p ignis-bench --target x86_64-unknown-linux-gnu
python3 scripts/remote.py exec -- make gpu-status
python3 scripts/remote.py exec -- make gpu-guard
python3 scripts/remote.py exec -- make start CUDA=1 UI=0 PROFILE=release \
  MODEL=qwen3.8-27b-16g BIND=127.0.0.1:18081 METRICS_BIND=127.0.0.1:19465 \
  IDLE_UNLOAD=0 LOOP_GUARD=off REQUEST_TIMEOUT=1800 STREAM_TIMEOUT=1800 VISION=0 \
  ARGS='--profile rtx50-16gb --speech-model off --diarizer-model off --reuse-prompt false --reuse-retained-device 0 --reuse-retained-host 0'
python3 scripts/remote.py exec -- make gpu-status
python3 scripts/remote.py exec -- target/x86_64-unknown-linux-gnu/release/ignis-bench \
  coding-agent --endpoint http://127.0.0.1:18081 --out .scratch/coding-agent-rerun/decode
python3 scripts/remote.py exec -- make gpu-status
python3 scripts/remote.py exec -- target/x86_64-unknown-linux-gnu/release/ignis-bench \
  coding-agent --capacity-only --endpoint http://127.0.0.1:18081 \
  --out .scratch/coding-agent-rerun/capacity
python3 scripts/remote.py exec -- make gpu-status
python3 scripts/remote.py exec -- target/x86_64-unknown-linux-gnu/release/ignis-bench \
  canary --endpoint http://127.0.0.1:18081 --out .scratch/coding-agent-rerun/canary.json
python3 scripts/remote.py exec -- make stop
```

Each status check must show only this daemon before running a leg. The baseline
used the stock Linux daemon support and checked its PID against GPU compute
processes before decode, capacity and canary. Output directories must be new.

Raw evidence is retained in both checkouts under
`.scratch/coding-agent-5080-2026-11-03/`: `decode/run.json`, `capacity/run.json`,
all requests and complete responses, metrics before/after, `canary.json`, server
logs, `gpu.csv`, `environment.json` (artifact and binary hashes), `commands.json`,
`results.json`, `inspection.json`, build/test logs and the scratch orchestration
script. `SHA256SUMS` covers every archived file. Pre-existing server logs were
preserved separately. The compressed archive is
`.scratch/coding-agent-5080-2026-11-03.tar.gz` in both checkouts.

Archive: 526,584 bytes, SHA-256
`75a92b175d68ec65418d338c44aba2c242dcfbc2f19f4df28aeff6210ff497f3`.

### Tests

- Remote Linux `cargo test --workspace --locked --target x86_64-unknown-linux-gnu`:
  **3,966 passed, 28 ignored**. Tests cover the first-round boundary, buffered tool
  output, EOS count convention, fixture hashes, cache/count/empty/truncation checks,
  and an HTTP runner replay that verifies independence and retained failed samples.
- The first full workspace run passed. A later full run after the EOS wording/test
  clarification failed the existing `the_token_never_appears_in_a_progress_line`
  download test: its immediate resume found the interrupted download's lock still
  held. The same complete command then passed without any source change. Both
  final logs are archived; the unrelated download test was not changed here.
- Laptop Python suite: **27 passed, 6 skipped** (Linux `/proc` daemon tests),
  including extractor tests. Frozen fixture regeneration reproduced all hashes.
- User guide build: **122 pages**, successful. Generated-reference check passed.
- The existing macOS `libc::O_DIRECT` dependency prevents the local Rust build;
  the full Rust workspace therefore ran on remote Linux. CUDA release build and
  the real-model runs above succeeded; HTTP mocks were not the final validation.

## Finding

On the measured 5080 configuration, the three fixed real coding-agent requests
decode at median **207.658 / 184.816 / 189.942 tok/s**, including the complete
server decode window. A separate exact 128K input completes prefill and generation.
These are observed results for this artifact, request set and protocol.

## Implications

`ignis-bench coding-agent` provides a repeatable baseline for later inference
changes: keep the fixtures, model, cache policy and generation protocol fixed,
and compare each task's full-window rate. The capacity pass remains a separate
check. The first report's short, synthetic requests and SSE timing define a
different workload and cannot be used as a direct speedup denominator.

## Limits and unknowns

This is **a performance baseline of three fixed real requests, not a complete
SWE-bench agent score**. Tools were not executed, tasks were not scored, and there
is no general code-quality or long-context answer-quality claim. Tool schemas
have the reconstruction limit described above. One server launch, one card and
three repetitions per request give no repeated-launch confidence interval or
SOL comparison. Greedy reproducibility does not establish behavior under stochastic
sampling. Two lanes were configured but only C1 was exercised. The capacity result
establishes 128K input plus generation, not the configured 262K maximum.

## Follow-ups

None tracked by this run.
