# Validation of the RTX 5080 scope cleanup

- Kind: experiment
- Status: current
- Observed: 2026-11-04
- Last verified: 2026-11-04
- Scope: fork scope / Linux serving / RTX 5080 Qwen3.8-27B trellis
- Related: [scope decision](../adr/0079-the-fork-serves-qwen38-27b-on-one-rtx-5080.md), [baseline](2026-11-03-coding-agent-5080-baseline.md), [benchmark protocol](../user/contributing/coding-agent-benchmark.md)
- Superseded by: none

## Question

How does the cleanup affect the fixed 5080 workload, Vision, tools and
service lifecycle? Which GPU checks fail?

## Evidence

The cleanup was developed in `codex/slim-5080`, based on
`9259978846d7db1a0161253138f6e34006dfbb64`. The measured source was the remote-sync
snapshot `388fd23c68546e32ec7bee66b4d5a489ab12cfc5` of the uncommitted worktree.
The benchmark baseline is unchanged. Subsequent edits covered documentation,
comments and commit-hook diagnostic pointers.

Raw results are in `.scratch/slim/` in the Linux cleanup checkout; copies of
the main records are in `.scratch/slim/evidence/` on the Laptop. These logs
are untracked. The run used the baseline's benchmark code, fixtures, protocol,
pinned trellis artifact and sidecar, release toolchain, two lanes, DFlash2
seven-token full head, disabled reuse, disabled loop guard and 1,800-second
deadlines. The server occupied 14,413 MiB, as in the baseline.
Runs were serial on the single RTX 5080.

Decode ran at 11:15:05–11:18:04 UTC. Every measured response was valid,
untruncated, ended in `tool_calls`, and had zero cached tokens.

| Request | Input tokens | Output tokens per repeat | Median decode tok/s | Change from baseline |
|---|---:|---:|---:|---:|
| Astropy | 19,954 | 876 | 207.454002 | −0.0981% |
| Django | 22,420 | 954 | 184.787909 | −0.0154% |
| SymPy | 16,717 | 1,268 | 189.966094 | +0.0128% |

Speculative `(rounds, drafted, accepted)` counts were respectively
`(161, 1127, 716)`, `(191, 1337, 764)` and `(264, 1848, 1005)`, identical to
the baseline. Median prefill times were 9.5554, 10.9123 and 8.0949 seconds.
The separate capacity check confirmed 131,072 input tokens, completed prefill
in 74.968 seconds and generated a nonempty answer. Canary passed.

HTTP smoke checks passed: all four Vision canaries returned the expected
semantic answers (47, red, three circles, and `error: config.yaml not found`);
the server produced a forced `read_file` call and processed its return;
Responses completed;
and a request after idle unload reloaded the model and received an answer.

| Check | Result |
|---|---|
| Laptop workspace Rust tests | 3,065 passed, 0 failed, 6 ignored |
| Linux workspace Rust tests | 3,066 passed, 0 failed, 6 ignored |
| Linux Python tests | 37 passed |
| Web tests and production build | 96 files / 1,147 tests passed; build passed |
| CUDA workspace `--all-targets`, including `attn-tap` | Passed |
| Trellis binding, layer oracle and lanes | Four GPU tests passed |
| Kernel CTest | 126 passed, four failed |

Four CTest failures reproduced their pre-cleanup failure signatures:
`ignis_kernel_gdn_gating_proj_tests`,
`ignis_hq_codec_test`, `ignis_kernel_kv_append_format_test` and
`ignis_kernel_gqa_workspace_capacity_test`. Before removal, 139 tests ran with
six failures; nine image-generation-only tests were removed, including the
other two failures. Numerical assertions in the remaining tests were unchanged.

Two more GPU checks failed:

- The trellis projection comparison needs the NVFP4 27B reference artifact.
  Only the trellis artifact is installed. Supply the control with
  `--reference-artifact`. The GPU profile fails when it is missing.
- The reload pool invariant fails on Linux: 4,858 cold pages become 5,242
  after reload, +7.9045%, exceeding the test's 1% limit. The same HTTP
  cold/unload/reload sequence was run on a freshly rebuilt, unchanged `main`
  and the cleanup binary with context 16,384, the same artifact/profile,
  DFlash2, host pool and disabled reuse. Both produced exactly those page
  counts and answered `OK`. Reload succeeds in both versions, but both
  violate the pool-size invariant.

## Finding

Decode medians differed from the baseline by at most 0.10%. The measured
service capabilities passed their smoke checks. The complete GPU profile
failed on the kernel, reference-artifact and reload checks listed above.

FP8 GEMM files moved out of the image-generation directory without byte
changes. The code includes NVFP4 drafter operators, reference implementations,
memory tiers and scheduling, along with historical artifact
tags and low-level Flash-Next dispatch code. The loader rejects unsupported
model families before loading them.

## Implications

The kernel failures and Linux reload accounting require separate fixes.
The projection comparison requires its NVFP4 control artifact to complete.

## Limits and unknowns

The timing results cover one run of three repetitions per task, which is
insufficient to establish a statistical speedup. Long-context answer quality
and service behavior beyond the listed smoke checks were not evaluated.

## Follow-ups

Use the failing tests and before/after logs to investigate kernel correctness
and Linux reload accounting. Compare future performance changes against the
frozen baseline.
