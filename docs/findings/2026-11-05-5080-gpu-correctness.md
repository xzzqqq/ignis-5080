# GPU correctness checks on the RTX 5080

- Kind: experiment
- Status: current
- Observed: 2026-11-05
- Last verified: 2026-11-05
- Scope: kernel / HQ codec and GDN dispatch; serving / Linux reload memory accounting
- Related: [cleanup validation](2026-11-04-5080-scope-cleanup-validation.md)
- Superseded by: none

## Question

Why do the four kernel checks and the reload invariant fail on the Linux 5080 baseline?

## Evidence

Base commit `d15be562`, branch `codex/fix-gpu-checks`, one RTX 5080, CUDA 13.0.88.
The unchanged pre-cleanup kernel build reproduces all four CTest failures.
Raw logs are under `.scratch/gpu-fixes/` in the remote repair checkout.

### HQ norm encoding

The codec writes an FP16 row norm to metadata bytes 0 and 1. In the failing build,
its PTX converts FP32 to FP16 and stores the low byte as bits. The generated SASS
instead inserts `F2I.U8.F16.TRUNC.NTZ` before the low-byte store: it converts the
half value to an integer. A row with norm 18.97324 should store `0x4cbe`, but stores
`0x4c12` (18 in the low byte). The high byte is correct.

`codec-before.ptx` and `codec-before.sass` preserve the disassembly. Instrumenting
the kernel with `printf` also makes the error disappear, so the diagnosis uses
the uninstrumented binary. Storing the norm as one aligned 16-bit word avoids
the incorrect conversion. No format or numerical tolerance changes.

Before the fix, `ignis_kernel_kv_append_format_test` reports 710 wrong norms,
up to 18.4753% relative error. `ignis_hq_codec_test` fails its normal and escalated
FP64-oracle comparisons. Both pass with the word store.

### GDN dispatch and workspace

The fused 27B norm candidate needs 240 resident CTAs. The planner allows two CTAs
per SM; the 5080's 84 SMs allow 168. The original fused selector does not test
legality and throws before a numerical check. The composed norm/control route
provides a legal fallback.

A second problem then appears at 1,024 projection tokens: the generic fallback
uses an unsplit reduction and fails the existing FP32 numerical criterion.
Choosing a smaller legal SplitK retains that criterion. SplitK can now change
inside a route interval; capacity must include interior tile boundaries, not just
the last width. The regression test checks every width through 4,097.

### GQA capacity assertion

On this GPU the old arena is 1,141,581,600 bytes and the replacement is
143,720,448: a saving of 997,861,152. The old split scratch depends on SM count,
so the 5090's minimum saving of one billion bytes is not portable. The replacement
is checked against its band/query/carry ceiling of 143,720,448 bytes instead.
Checks for context-independent workspace and coverage of every call remain.

### Reload accounting

The baseline adds 536,489,984 bytes to free memory on reload, based on historical
Windows/5090 process residency. Linux now queries NVML's compute-process accounting
for its own PID before a load. It adds only the resident bytes attributed to that
process. Failure to read the counter refuses the load. The cold-load reservation
constants are unchanged; this fix changes the reload add-back. The process retains
299,892,736 bytes after teardown. Three consecutive reloads reserve 4,845 pages against
4,858 cold (0.27% apart), instead of the baseline's 5,242. Free memory with the
model loaded is 1,509,818,368 bytes cold and 1,518,206,976 after reload.

### Final checks

| Check | Result |
|---|---|
| Kernel CTest | 130 passed, zero failed; 233.43 seconds |
| Laptop workspace CPU suite | 3,065 passed, zero failed, six ignored |
| Linux `make ci CUDA=1 UI=0` | Passed on rerun: 3,066 Rust tests and 1,147 web tests; six Rust tests ignored |
| Python tooling tests | 31 passed, six skipped |
| Web production build | Passed with Node 24.22.0 |
| CUDA workspace all targets, `ignis-server/cuda,ignis-server/attn-tap` | Passed |
| NVML process/allocation accounting | Passed |
| Cold load and three reloads | Passed, unchanged 1% tolerance |
| Trellis binding | Two tests passed |
| Trellis layer f64 oracle | Passed |
| Trellis lanes and speculative/plain agreement | Passed |
| Trellis/NVFP4 projection comparison | Passed, cosine floor 0.9 unchanged |
| Vendor manifest and patch tests | Passed |

The first Linux CI attempt failed in the unchanged download test
`a_server_that_ignores_the_range_restarts_the_transfer_cleanly`: its second fetch
reported that the entry lock was still held. The log is
`.scratch/gpu-fixes/make-ci-first.log`. The full rerun passed, including this
test; the transient failure has not been diagnosed.
Frontend checks use the installed Node 24.22.0; the machine's default Node 19
cannot run the pinned TypeScript version.

The projection control is the historical revision
`e961b419b672e183aa55df8c4b975abc82006e8a` of
`gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer`. Its 19,406,942,468-byte artifact
matches SHA-256 `abb1e120d5f1f32d61689604d238227ff579ab76cbd9319628f3b3904fffd9af`.
It was downloaded through the HF mirror after the primary endpoint timed out.
The comparison runs NVFP4 reference products on the CPU and one Trellis matrix
at a time on the GPU. Observed projection cosines include 0.92774 for layer 0's
MLP down projection and 0.99673 for the output head.

CTest and each Rust GPU stage were run serially, using `gpu_profile.stage` and
its exclusive marker for the Rust stages. The projection ran separately after
the download; this was not one invocation of the full-profile command.

### Frozen coding-agent benchmark after the fixes

Run on 2026-11-05, 15:06:59–15:10:29 Asia/Shanghai, followed by capacity and
canary. Release server and bench built from `d15be562` plus the repair worktree;
the measured Git tree is `0d56efd65f31d03bef4be7b0a3e969f112eccfa2`.
The source patch and binary hashes are archived with the results.

The [fixed benchmark protocol](../user/contributing/coding-agent-benchmark.md)
is unchanged: one warmup and three measured responses per task, C1, two configured
lanes, DFlash2/7, 262,144 context, chunk 1,024, reuse and loop guard off. Every
recorded configuration value matches the original baseline. The card retains its
360 W power limit, PCIe Gen 5 x16 under load and 14,413 MiB serving footprint.

| Task | Input tokens | Output tokens: baseline → repaired | Baseline tok/s | Repaired median tok/s | Change | Median prefill s | Median HTTP s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Astropy | 19,954 | 876 → 913 | 207.657750 | 186.466100 | −10.2051% | 9.5241 | 14.4469 |
| Django | 22,420 | 954 → 2,319 | 184.816402 | 182.072210 | −1.4848% | 10.8341 | 23.5989 |
| SymPy | 16,717 | 1,268 → 1,379 | 189.941801 | 213.895607 | +12.6111% | 8.0443 | 14.5170 |

All nine measured responses are valid, nonempty, untruncated, finish with
`tool_calls`, and report zero cached tokens. Each task's warmup and three responses
are identical after removing generated tool-call IDs; all tool arguments parse as
JSON. The emitted tools are respectively `bash`, `bash`, and `str_replace_editor`.
No generated command was executed and no task solution was scored.

| Task | Repaired rounds / drafted / accepted | Mean round ms: baseline → repaired |
|---|---|---:|
| Astropy | 187 / 1,309 / 727 | 26.2017 → 26.1836 |
| Django | 472 / 3,304 / 1,848 | 27.0255 → 26.9845 |
| SymPy | 255 / 1,785 / 1,125 | 25.2869 → 25.2826 |

Counters repeat identically across each task's four responses. Round times are
medians of each sample's decode time divided by its round count. They change by
less than 0.2%, while generated text, response length and speculative acceptance
change. These measurements therefore support a changed generation path as the
main explanation of the rate differences; they do not isolate an individual fix.
Django's longer output increases total response latency from 16.0942 to 23.5989 s.

The independent capacity check prefills exactly 131,072 tokens in 74.8356 s and
emits one output token, with no cache hits or truncation. Canary passes all four
sanity/determinism checks; two reach an answer and two remain thinking-only under
its existing short budget. The daemon is stopped and the GPU is free afterward.

Raw requests, responses, server logs, GPU samples, commands and analysis are in
`.scratch/gpu-fixes/benchmark/` remotely and copied to the Laptop under
`.scratch/gpu-fixes/benchmark/evidence/benchmark/`. The Laptop archive is
`.scratch/gpu-fixes/benchmark.tar.gz`, SHA-256
`b18f58472c0f2ffc02380c47fab6be33fd5c773c02f6680ae0596788c5eb9135`.
This is one launch and three repeats per task, not a repeated-launch confidence
interval. The designated optimization baseline remains unchanged.

## Finding

The HQ failures expose corrupted metadata from generated machine code. The GDN
failures expose GPU-dependent dispatch and capacity assumptions. The GQA failure
is a hardware-dependent savings assertion, rather than an arena overrun.

## Implications

The HQ fix changes encoded values. Historical output and quality measurements
from an affected compiler cannot establish the corrected build's exact outputs.
Keep the frozen benchmark inputs, but label measurements with the build.

## Limits and unknowns

The HQ compiler failure is established for this CUDA 13.0.88/SM120a build; other
compiler releases and the original Windows build were not tested. These operator
checks do not replace end-to-end quality evaluation.

## Follow-ups

Keep the verified reference artifact available for future projection checks.
Evaluate task-level quality on the corrected codec. Future performance comparisons
should preserve the frozen inputs and report changed outputs and acceptance beside
rates, as this run does.
