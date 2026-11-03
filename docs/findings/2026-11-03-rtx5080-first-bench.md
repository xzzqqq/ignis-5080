# First Qwen3.8-27B bench on the RTX 5080

- Kind: experiment
- Status: current
- Observed: 2026-11-03
- Last verified: 2026-11-03
- Scope: serving / Qwen3.8-27B trellis on RTX 5080, Linux build and basic HTTP bench
- Related: [remote development](../user/contributing/remote-dev.md), [bench protocol](../agents/testing/bench.md), [ADR 0063](../adr/0063-the-profile-follows-the-card-class.md), [the earlier 16 GB speed estimate](2026-10-24-27b-16gb-speed-estimate.md)
- Superseded by: none

## Question

Does the fork build and serve its 16 GB Qwen3.8-27B artifact on a real RTX 5080, and what does the existing
`small-card` bench measure before any inference changes?

## Evidence

Source: `cafa904e725a43d04bf6b5142a4e354b8ebe28d5` (Ignis 0.9.5), with a clean remote working tree.
No Rust, CUDA, Makefile or model changes were needed. The measurement ran on 2026-11-03 from
20:37:34 to 20:40:21 Asia/Shanghai, followed by the canary suite.

### Machine and build

| Component | Observed value |
|---|---|
| GPU | NVIDIA GeForce RTX 5080, 16,303 MiB, 360 W power limit |
| PCIe during decode | Gen 5 x16, also the reported maximum |
| Driver | 580.105.08 |
| Host | Intel Core i9-13900K, 125 GiB RAM, Ubuntu 22.04 / Linux 5.19 |
| Toolchain | Rust 1.99.1, CUDA 13.0.88, CMake 4.4.5, GCC 11.3.0 |
| Build | Linux x86_64, release, `cuda`, SM120a |

The existing CUDA 13 environment lacked `nvml.h` and `nvtx3/`. Copies from the machine's CUDA 12.5
headers were placed in `.scratch/toolchain/include` and supplied through `CPATH`. A private CUDA
directory made `lib64` point to the environment's `targets/x86_64-linux/lib`, as the Rust build expects.
The system CUDA installation was unchanged. CMake 4.4.5 was already available outside the default PATH;
the default CMake 3.22.1 was below the project's 3.28 minimum. Rust was installed without changing shell
startup files. Dependency downloads used a task-local Cargo mirror configuration with the existing lockfile.

The existing model file matched the catalog exactly:

- Artifact: `qwen3_8_27b_trellis_sc30-v2.ninfer`, 13,379,877,124 bytes;
  SHA-256 `2c4c9ee4d6b25c08da6c2d85840126e73d120564e230f926b6230b5b95ee8b3c`.
- The first start was refused because its sidecar was absent. The 309,495-byte
  `.ninfer.conversion.json` was downloaded from `gpillon/Qwen3.8-27B-trellis-3bpw-ignis` at the catalog's
  revision `f5bbd2cc14c9ca573ef5553c100ac69410f1fe58`, then checked against SHA-256
  `210ab54bfe8b04e5fc4337ee3755c340ea3f21239aa625c505aa2406645a340d`.

All remote operations went through `python3 scripts/remote.py exec -- ...`. The CUDA leaf was built with
`bash kernel/build.sh`, the server with `make build CUDA=1 UI=0 PROFILE=release`, and `ignis-bench` with
`cargo build --locked --release -p ignis-bench --target x86_64-unknown-linux-gnu`.
`mk/cuda-build-check.sh` confirmed that the server was a CUDA build.

Linux's stock daemon hooks remain unimplemented. This run supplied task-local `SERVER_START` and
`SERVER_STOP` hooks to `make start` / `make stop`, keeping the daemon's PID and process start time.
Each GPU leg had a `make gpu-status` check; the initial legs also passed `make gpu-guard`.
The bench verified that its daemon was the only GPU compute process. The server was stopped afterward,
and a final check showed 0 MiB used and no compute process.

### Serving shape and measurement

`rtx50-16gb`, 262,144 context, two decode lanes, 1,024-token prefill chunks, hq-e8-2b KV,
DFlash2 with seven drafts and the full proposal head. Loop guard and idle unload were off.
The profile's vision support remained enabled with a 4,096-token envelope and host residency;
all measured requests were text. `VISION=0` on make omits its enabling flag and does not override
the profile's `vision.enabled: true`. UI, speech and diarization were off.

The load adapted nothing. Its KV pool had 4,851 pages / 310,464 tokens. The VRAM plan's total was
15,347,997,868 bytes against a 15,348,203,520-byte budget, with no oversubscription. The sampled
device memory usage was 14,415 MiB throughout the bench; the plan and NVML reading are different
accounting views and should not be substituted for each other.

Against the fresh daemon, exactly one invocation:

```sh
target/x86_64-unknown-linux-gnu/release/ignis-bench small-card \
  --endpoint http://127.0.0.1:18080 \
  --cells 8192,65536 --samples 3 --decode-tokens 256 \
  --out .scratch/bench-5080-2026-11-03/small-card.json
```

The preset sends four short coding/math prompts with thinking off, then three synthetic agent turns
with thinking on, sequentially. Its decode rate is `(completion_tokens - 1) / time_after_first_token`.
Each TTFT cell has one warmup and three measured prompts. Both cells passed cold-prefix validation,
with zero void samples and no report notes.

| Measurement | Result |
|---|---:|
| Coding decode, median of four request rates | 285.06 tok/s |
| Coding decode, pooled rate | 257.11 tok/s |
| Agent decode, median of three request rates | 242.77 tok/s |
| Agent decode, pooled rate | 241.85 tok/s |
| 8,192-token cold TTFT, median | 3,960.24 ms |
| 65,536-token cold TTFT, median | 34,359.06 ms |

The measured TTFT samples were 3,955.84 / 3,960.24 / 3,969.50 ms and
34,284.03 / 34,359.06 / 34,392.69 ms, respectively. Across the decode sets, 685 of 1,038 draft tokens
were accepted over 151 rounds: 65.99% acceptance, 4.54 accepted drafts per round.

The one-second GPU sampler recorded 168 samples: peak 362.69 W, temperature 47–83 °C, and SM clock
2,610–2,872 MHz. These span decode and prefill, with prefill occupying most of the run; their aggregate
utilization is not a decode utilization measurement.

### Checks

- Linux `cargo test --workspace --locked --target x86_64-unknown-linux-gnu`: **3,958 passed, 28 ignored**
  on the complete rerun. The first run failed only `pings_do_not_keep_an_idle_socket_open` in
  `responses_socket.rs`, at `opened.elapsed() >= IDLE` (300 ms). Its isolated rerun and the full rerun
  passed without a source change. Both logs are retained; the timing-sensitive failure was not fixed here.
- CUDA substrate smoke: `ctest --test-dir kernel/build --output-on-failure -R '^ignis_kernel_op_tests$'`,
  **1/1 passed** on the GPU.
- `ignis-bench canary` after the bench: **4/4 sane and greedy-deterministic**, self-consistency PASS.
  Two canaries reached an answer; two exhausted their budget while thinking. This is the harness's
  sanity result, not a claim that every canary answered its question.
- Laptop remote-entry Python tests: **5/5 passed**. The attempted macOS Rust workspace build failed
  because `libc::O_DIRECT` is unavailable; the workspace checks above ran on Linux instead.

Raw evidence is untracked under `.scratch/bench-5080-2026-11-03/` in both checkouts:
`small-card.json`, `canary.json`, `server.log`, `server-command.json`, `make-arguments.json`,
`gpu.csv`, `metrics-after.prom`, `environment.txt`, the build/test logs, and the task-local
`env.sh`, `serve.py`, `run-bench.py`. Failed-start logs are preserved separately. The final debug test
artifacts were removed to recover about 71 GiB; release binaries and CUDA build artifacts remain.

## Finding

The unchanged fork builds and serves the catalog's 16 GB trellis artifact on this RTX 5080. Its default
16 GB context and lane count fit without adaptation. The existing HTTP bench and canary complete,
providing a first measured baseline on the actual target card.

## Implications

This baseline can anchor work on the fork when the same model, serving shape and prompts are used.
The setup requires the catalog sidecar as well as the weights, a complete CUDA header set, and a Linux
daemon hook. Those environment requirements were sufficient here; no inference change was required.

## Limits and unknowns

One measured server launch; no repeated-launch confidence interval, thermal stabilization protocol,
reference engine, or SOL comparison. The decode sets have short prompts and variable output lengths:
17–122 tokens for coding and 44–256 for agent turns. Their rates are not long-context decode rates and
cannot directly validate the earlier 8K/64K/253K speed estimate.

Two lanes were configured, but the bench issued sequential requests; concurrent throughput was not
measured. Loading a 262K context does not test a 262K request. Vision was loaded but no image request
was sent. This run did not execute the full GPU profile or quality gates.

## Follow-ups

None tracked by this run.
