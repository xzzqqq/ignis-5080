# 05 - first light, gates and simulation: the 16 GB artifact served in ignis, its quality, 16 GB capacity on the 5090, the speed estimate

GitHub: #339 (Part A, 05a: first light) and #340 (Part B, 05b: gates and simulation), master #334.

Decided 2026-10-21: the owner approved this spec and vetoed none of the *(proposed)* items, so every one is decided.

Phase 1 has no 16 GB card (the owner, 2026-10-21). This spec is what phase 1 measures instead, and how:
- **quality**, end to end in ignis, against the owner's gates;
- **capacity**, simulated exactly on the 5090;
- **speed**, as an estimate stated as one, with a bench users can run to report real numbers.

ADRs: 0060 (the gates), 0063 (adaptation), 0003/0014 (gating practice), 0022 (BF16 as oracle).

## Part A (ticket 05a) - first light

The point where specs 01-04 meet: the 16 GB artifact served end to end by ignis on the 5090, under a 16 GiB budget.

**What it wires:**
- the 27B program binds a trellis body, fused planes or the split parts the repack carries (spec 01), by object
  presence;
- every layer's linears dispatch to spec 02's ops by tensor format;
- the token embedding is read from host memory (spec 03);
- the load runs the `rtx50-16gb` profile with its lanes and adaptation (spec 04).

**Acceptance (A):**

A1. The 16 GB artifact loads on the 5090 with `--vram-budget-bytes 17179869184` under the `rtx50-16gb` profile. The
    VRAM plan and any adaptation are logged, and the plan's lines match spec 04's formulas.
A2. It serves the coding canaries greedy, end to end: chunked prefill, then decode with DFlash2 at 2 lanes, the 128K
    leg included. Outputs are coherent and there are no errors. Tokens per round and round time are reported.
A3. A sanity KLD on two of the 2026-09-24 windows, engine path, reported beside M1's weights-only numbers for the same
    windows. This is not the gate (Part B); it catches a wiring error early.
A4. The 32 GB NVFP4 artifact still serves exactly as before: greedy outputs on the canaries are identical.
A5. `cargo test` workspace-wide; GPU tests green on a free 5090.

### As built: the wiring (2026-10-23)

- **The body is the head's.** `text/output_head` in a trellis format makes the body trellis, on both sides of the
  ABI: `ignis_artifact::body_27b` reads the directory, and `model_scope_27b_of` gives the binder and the load's
  descriptors spec 01's trellis text scope (`text_scope_27b_trellis`, K from the directory); the leaf's binder
  (`kernel/src/model.cu`) reads the head's qtype and binds the whole body under that schema. `ModelScope` stays the
  modules a load asks for. A tensor of the other body fails naming itself ("a load binds one body, never a mix").
  `fp8_exceptions` converts BF16 tensors only, so on the trellis body, whose `attention/output` of layers 3 and 7
  and `gdn/output` of layer 4 share the exception names, it is a no-op.
- **Fused trellis planes are not built.** What it wires asks for "fused planes or the split parts, by object
  presence"; only the split parts bind, and the body, not each object's presence, picks the schema. The trellis
  schema names the parts (`gdn/in_proj_qkv` and `in_proj_z`, `attention/q_proj`, `k_proj` and `v_proj`,
  `mlp/gate_proj` and `up_proj`), so a fused trellis plane in their place fails as the part it lacks, and the served
  schema's `bind()` refuses any trellis code by name. Spec 01 keeps fused planes, from our own conversion, as the
  fallback if spec 05b's gates fail: that fallback would need a schema arm of its own, which does not exist.
- **ABI.** Six leaf-only qtype codes, `IGNIS_QTYPE_TRELLIS_MUL1_K2` .. `_K6` = 64 .. 69, and
  `IGNIS_LAYOUT_TRELLIS_TILE16` = 64, past `ninfer::QType` / `QuantLayout` (vendored, untouched): a trellis tensor
  binds into an `ignis_trellis_weight`, never a `ninfer::Weight`. Its `qhigh` and `scales` are the record's `suh` and
  `svh` (the materializer's view names them); the binder checks both against the tiles' length and the record's
  bytes against its K.
- **Dispatch** (`kernel/src/trellis_body.h`): the three GDN bodies and the two GQA ones call the op in place of the
  vendored NVFP4 ops on a trellis load -- `in_proj_qkv` / `in_proj_z` into `qkv` / `zbuf`, the output and down
  projections as the op's residual add, the gate and up matrices as its SwiGLU, `k_proj` / `v_proj` into key and
  value, `q_proj` into a `[T][12288]` buffer whose heads' halves two `cudaMemcpy2DAsync` move into the query and the
  gate. The verify GDN body writes its projection into the conv record and runs the vendored
  `gdn_projected_conv_record_launch` over it. Every head site -- the four prefill ones, the decode and verify rounds,
  and the drafter's full head -- goes through `output_head`, the shortlist head staying the served Q4 one. The
  test-only `kA16Only` mode is the NVFP4 arm's alone.
- **The `trellis_workspace` plan line** (ADR 0030 amended, `ignis_model_reservations::trellis_workspace_bytes`,
  `VramLines::trellis_workspace`, the plan event, `/metrics`, the Monitor): one buffer at a stable address on the
  model's one stream (the drafter's calls run there too), zeroed once at load, `ignis_trellis_prepare` run by the
  plan and the load. Its size is the op's own answer for the widest call: the body's shapes at 1..64 tokens (a short
  prompt or a span's last chunk takes the decode route) and at the chunk, the head's at a round's columns across the
  lanes (`lanes x (k + 1)`) or one span-logits block (32). 36,225,280 bytes at chunk 1,024 with `rtx50-16gb`'s two
  lanes and seven drafts; 0 on the served body.
  - The head is asked on the op's decode route alone, so its widest call must stay on that route (64 tokens). Every
    head call a trellis load makes is bounded at compile time: a decode round's lanes (at most 8), a verify round's
    `lanes x (k + 1)` (at most 8 x 8 = 64), the drafter's `lanes x k`, a span-logits block (32), and prefill's last
    token (1). `static_assert`s in `kernel/src/model.cu` pin the verify round's and the block's against
    `IGNIS_TRELLIS_DECODE_MAX_TOKENS`. A load whose head would still be wider is refused at plan time, naming why,
    and never fails mid-decode. Before this, the plan clamped the head's width silently.
- **The plan's trellis lines** (what A1 holds the plan to). Spec 04's formulas are `PIANO-VRAM.md`'s table ("Da dove
  vengono le cifre", fitted to the served body's plans). Its `workspace` is 0.34 GB of chunk scratch plus 92 MB of
  DFlash2 append; the hq prompt planes term is gone since ADR 0061. The code those formulas summarize is
  `kernel/src/model.cu`'s `compute_program_scratch_bytes`: `program_outer_scratch_bytes` plus the wider of
  `gqa_layer_scratch_bytes` and `gdn_layer_scratch_bytes`. The trellis body changes three lines. In closed form, with:
  - C the chunk, L the lanes, k the draft window, n the context;
  - b(x) and w(x) an arena buffer of x BF16 or 4-byte values, rounded up to 256 bytes;
  - the 27B's widths: H 5,120, Q 6,144, K 1,024, F 17,408, V 248,320, and for GDN 10,240 conv channels, 2,048 key
    width, 6,144 state rows and 48 value heads;

  the terms are:
  - outer(T, B) = w(TB) + 2 b(H TB) + b(H N) + b(V B) + w(1), where N = TB on a verify round and B otherwise;
  - GQA(T, B) = 2 b(H TB) + 4 b(Q TB) + 3 b(K TB) + w(TB) + w(B) + b(F TB) + A + b(2Q TB). A is the attention's
    query (`ignis_gqa_attention_capacity_bytes` at n keys). b(2Q TB) is `q_proj`'s interleaved q | gate before its
    split;
  - GDN(T, B) = 2 b(H TB) + 2 b(10,240 TB) + 2 b(2,048 TB) + 4 b(6,144 TB) + 2 w(48 TB) + b(F TB) + G + D, where G
    and D are the gating's and the recurrence's queries.

  On the NVFP4 body, GQA and GDN add their projections' workspace queries (`attn_input_proj` or `gdn_input_proj`,
  `linear_add` twice, `linear_swiglu`) and have no q | gate buffer. On the trellis body every projection is the op's,
  and the op's workspace is its own line. The three lines are then:
  - `workspace` = outer(C, 1) + max(GQA(C, 1), GDN(C, 1)) + P(C) with DFlash2. P(C) = b(5H c) + w(c) + 2 w(1) +
    2 b(H c) + b(6,144 c) + 3 b(1,024 c), with c = min(C, 2,048), is the drafter's context append, the same on both
    bodies (`PIANO-VRAM.md`'s 92 MB). There is no `locate` readout room: the trellis artifact has no `locate`
    calibration;
  - `decode_graph` = outer(k + 1, L) + max(GQA(k + 1, L), GDN(k + 1, L)) + 8L;
  - `trellis_workspace` = the op's layout at the load's widest call, 16,640 + ⌈2 m T in⌉ + ⌈4 m (in / 128) T⌉,
    plus ⌈4 s T m out⌉ on the decode route. ⌈·⌉ rounds up to 256 bytes, m is the SwiGLU pair's 2 on the decode
    route and 1 otherwise, and s is the decode route's split. The maximum runs over the body's seven shapes at T in
    1..64 and at C, and over the head's at T in 1..max(L (k + 1), 32).

  At the 16 GiB load (C 1,024, L 2, k 7, n 262,144, hq-e8-2b, DFlash2):
  - `workspace` 395,898,880 = outer 21,482,752 + GQA 282,136,832 (A 143,720,448; the GDN layer's 232,325,120 is the
    narrower) + P 92,279,296;
  - `decode_graph` 20,621,328 = outer 1,485,312 + GQA 19,136,000 (A 16,972,800; GDN 2,512,896) + 16;
  - `trellis_workspace` 36,225,280 = 16,640 + 2 x 1,024 x 17,408 + 4 x 136 x 1,024: `mlp/down`, the widest `in`,
    on the prefill route at the chunk. No decode-route call is wider, the head's at 32 columns included.

  `test_model_load_trellis` recomputes `workspace` (without P, under the verify-only backend) and `decode_graph`
  from these forms over the vendored queries, prints the terms, and checks the plan equals them. It holds
  `trellis_workspace` to the op's own maximum. `trellis_16g_gpu.rs` pins the three numbers on the server's load.
- **A fix in spec 02's op.** Its workspace's arrival counters sat in a room sized by the call's own output blocks,
  with the rotated activations after it, so a narrow call (a down projection) left a wider call's counters nonzero
  and that call reduced its splits early: every layer of the trellis body ran 0.17-0.86 from the oracle until the
  room got one size (`kArrivalBlocks`, `out` at most `IGNIS_TRELLIS_MAX_OUT` = 524,288, workspace geometry and not a
  serving threshold; +16 KB of workspace). `test_trellis_linear`'s
  `shared_workspace_arm` runs a load's calls of five shapes through one workspace, each against its own fresh one.
- **Tests and what they measured on the 5090:**
  - `kernel/tests/test_model_load_trellis.cpp` (the binder, the plan line, and the closed forms of `workspace` and
    `decode_graph`) and `test_trellis_body.cu` (the q/gate split, `query_gate`, the head, the epilogues); both fail
    on the split's halves swapped.
  - `crates/core/tests/trellis_projections_gpu.rs`: every split part is the served plane's rows in spec 01's order,
    cosine 0.928-0.997 per block, the q/gate halves the other way round 0.005 or less.
  - `crates/core/tests/trellis_layers_gpu.rs`: GDN layer 0 and GQA layers 3 and 27, one and four tokens, against the
    f64 oracle over the same decoded weights: relative L2 0.0018-0.0047 (bound 0.01); the split's halves swapped:
    0.78-1.0.
  - `crates/core/tests/trellis_logits_gpu.rs`: 768 positions of code, frozen as a fixture
    (`tests/fixtures/trellis_logits/code_span.txt`): top-1 agreement with the served program 683 of 768 (0.889),
    mean KL(served || trellis) 0.071 nats (floor 0.7, an error detector).
  - `crates/core/tests/trellis_lanes_gpu.rs`: two lanes. The layer bodies a width of two runs are leaf-internal,
    reached only through a whole round, so they are held by the tokens a round commits, as the NVFP4 body's are,
    and not by the f64 oracle. Two prompts, 64 greedy tokens each:
    - in width-2 rounds from the decode graph, each lane is identical to its width-1 text;
    - in width-2 verify rounds with the drafter at window 7 (16 columns, the GDN record arm and the GQA batch, 5.36
      committed tokens per full-window round), each lane is identical too;
    - the width-2 rounds run eagerly, plain and verify, are bit for bit their graphs'.
  - `crates/server/tests/trellis_16g_gpu.rs`: the server's load path, `--profile rtx50-16gb --vram-budget-bytes
    17179869184`:
    - the plan: nothing adapted (asserted), 262,144 context, two lanes, a 509,184-token KV pool, weights
      10,907,958,020 bytes, and its three trellis lines pinned;
    - greedy code is coherent and deterministic, at 7.68 DFlash2 tokens per round;
    - the two prompts at once, both decoding before either is done, give each its own sequential text.
  - `crates/artifact/tests/trellis_27b.rs`: the model scope binds the real artifact by its body at spec 01's device
    bytes; CPU tests for the codes, the body detection, the fp8 guard and the structs' layouts on both sides.
- **Left to the acceptance runs:** A1's plan log through `make start`, A2's canaries and 128K leg, A3's sanity KLD,
  A4's canary identity on the served artifact.

### As built: acceptance (2026-10-23)

Branch `339-first-light` at 60f4447b, release build, on the RTX 5090 (PCIe Gen 3, ~2 GB held by the desktop). The
evidence is in `.scratch/339/acceptance/` (untracked); the finding is
[2026-10-23-27b-16gb-first-light](../../findings/2026-10-23-27b-16gb-first-light.md).

- **A1: pass.** `make start ARTIFACT=F:/ai/models/Qwen3.8-27B-trellis-3bpw-ignis/qwen3_8_27b_trellis_sc30-v2.ninfer
  VRAM_BUDGET=17179869184 LOOP_GUARD=off IDLE_UNLOAD=0 ARGS="--profile rtx50-16gb"` (`a1-make-config.txt`,
  `trellis-server-load.log`):
  - nothing adapted: 262,144 context, two lanes, chunk 1,024, `dflash2 draft_tokens=7 draft_head=full`
    (`draft_head auto -> full`), vision off, rope none;
  - the three trellis lines equal their closed forms above: `workspace` 395,898,880, `trellis_workspace` 36,225,280,
    `decode_graph` 20,621,328;
  - the lane-scaled lines are spec 04's per-lane costs, the 8-lane NVFP4 plan's divided by four: `lane_state`
    393,760,768 (196,880,384 a lane: GDN fp32 151 MB, conv 3 MB, penalties 1 MB, DFlash2 window 40 MiB) and
    `hq_residual_window` 71,303,296 (35,651,648 a lane). `sampling` is 1,156,768, a quarter of the 8-lane 4,623,344
    plus 932 bytes that do not scale. The round buffers, `decode_graph` + `verify_round` 38,034,808 +
    `drafter_round` 44,673,544, are 103.3 MB against the formula's 2 x ~35 MB + 34 MB;
  - `cuda_context` 452,595,712, `residual` 124,474,800, `media` 0, `retained_slots` 0, `weights` 10,907,958,020;
  - the KV pool is 4,692,672,512 bytes, 7,956 pages, 509,184 tokens: one 262,144-token sequence and the 65,536-token
    spare reservation fit, so the ladder never ran;
  - total 17,179,375,716 of 17,179,869,184. On the release build every line equals the one `trellis_16g_gpu.rs`
    measured on the server's load path, `residual` included (free at start: 31,642,738,688).
- **A2: pass** on the spec's terms. Two lanes repeat their sequential text up to one near-tie. Same server.
  - Canaries (`ignis-bench canary`, `trellis-canary.json`): 4 of 4 sane, deterministic and coherent (`v` becomes
    `[1,2,3]`, 2 * 3 + 4 is 10). One lane: 4.41 tokens per round over 116 rounds, 12.83 ms a round.
  - Two lanes: two `canary` processes at once, four runs, 64 requests (`trellis-canary-pair*`). In every overlap both
    requests' `ignis.request.ttft` come before either's `done` (run 1: 18:55:37.883 and .941, then 38.123 and
    38.138). 4.27 tokens per round over 960 rounds, 16.52 ms a width-2 round: ~517 tokens/s across the two lanes,
    against ~344 for one.
    - 56 of 64 equal their sequential text. The 8 that differ are every `math-greedy` served on lane 7 in a width-2
      round: one token of reasoning ("on last line. Could" against "on the last line. We can"), same answer.
    - At that position the trellis program's two candidates are one BF16 step apart: logit 28.125 against 28.0,
      p 0.53 / 0.47 (`a2-neartie-probe.txt`). A near-tie, which `crates/core/tests/support/near_tie.rs` records as
      not route-invariant. Run the same way on its 8-lane load (A4), the NVFP4 body differs from its width-1 text on
      all 48 requests.
  - The 128K leg (`soup_ttft.py`): word soup landed exactly on 131,072 tokens through `/v1/tokenize`, one warm-up
    and two samples, each cold (`cached_tokens` 0) and coherent:
    - TTFT 42,984 / 43,136 / 43,216 ms (~3,040 tokens/s);
    - 4.47 tokens per round over 43 rounds, 19.33 ms a round.
  - `ignis-bench ttft --cells 131072` cannot do this leg. Its filler generator tokenizes the whole message again after
    every word it appends, and one 129K-token encode costs 0.49 s here, so a prompt would take ~8 h. It was stopped
    after 5.5 min, before any request.
  - A coding request of 129,992 tokens (`long_code.py`: `crates/bench/src/ttft.rs`, then four server sources), greedy,
    thinking off, asks what `nonce(27)` returns:
    - the answer is coherent, cold, at a TTFT of 42.5 s, and wrong in one letter: "`a027`", with the right mechanism
      (27 % 26 = 1, zero-padded `027`) but index 1 read as `a`;
    - the trellis body says the same at 23,996 tokens, and the served NVFP4 artifact says "`a027`" at both lengths.
      This is the model answering without thinking, not the body;
    - 3.56 tokens per round, 19.31 ms a round.
  - `/metrics` after the leg: 1,151 rounds, 7,532 drafted, 3,772 accepted (4.28 tokens per round). Acceptance by
    position: 87.7, 69.1, 54.9, 41.7, 33.5, 31.0, 26.1%.
  - The log's only error is the start-up probe's 503: `make start`'s `/v1/models` landed between `model.loaded` and
    `process.ready`. Besides the canaries' reasoning-only notes there is one warning,
    `decode_graph_capture_incomplete` with `ready` 2 at two lanes and an empty error.
    `crates/runtime/src/cuda_leaf.rs` compares the widths ready to 8, not to the load's lanes (since #336), so the
    warning is spurious.
- **A3: pass.** `span_logits` over windows 0 and 1 (`crates/server/src/api.rs` parts 0 and 1, 4,096 tokens), chunk
  1,024, scored by `.scratch/kld-2026-09-24/ref_kld.py` against BF16 (`acc339-trellis.json`). KLD / top-1 / NLL:

  | window | trellis, hq-e8-2b | trellis, BF16 KV | M1, weights only | NVFP4, BF16 KV | NVFP4, hq-e8-2b |
  |---|---|---|---|---|---|
  | 0 | 0.0534 / 90.80% / 1.2357 | 0.0421 / 91.72% / 1.2273 | 0.0420 / 91.60% / 1.2276 | 0.0553 / 91.70% / 1.2365 | 0.0666 / 90.89% / 1.2468 |
  | 1 | 0.0304 / 93.82% / 0.9748 | 0.0270 / 94.53% / 0.9712 | 0.0271 / 94.51% / 0.9708 | 0.0363 / 93.65% / 0.9823 | 0.0401 / 93.43% / 0.9848 |

  M1 is `RISULTATI-M1.md`'s per-window list (its order is `windows_meta.json`'s). The NVFP4 columns are the served
  engine path at chunk 1,024: 2026-09-24's `bf16_1024` and #341's `c1024`. With BF16 KV the trellis engine path is M1
  within 0.0001: the leaf's numerics add nothing visible. The codec adds +0.0113 / +0.0034, as it adds +0.0113 /
  +0.0038 to NVFP4 on the same windows. Both windows: 0.0419 against NVFP4's 0.0534.
- **A4: pass.** NVFP4 from this build: `make start IDLE_UNLOAD=0 MAX_CONTEXT=786432
  ARTIFACT=F:/ai/q38/ninfer-models/qwen3_8_27b_nvfp4full-v2.ninfer` (main's `./models` link resolved):
  - canaries identical to main 05ccde09's baseline (text and reasoning, both runs of each);
  - `ignis-bench oracle compare`: 100%, 265 of 265 recorded tokens (`--first-n 128`);
  - the plan's lines equal the baseline's but the pool's, which takes the 904,663,040 bytes more free at start
    (31,651,422,208 against 30,746,759,168): 15,529 pages and 993,856 tokens against 13,995 and 895,680.
    `trellis_workspace` is 0; `adapted` is empty because the context was named.
  - `MAX_CONTEXT=786432` is the context the baseline adapted to. With the baseline's own command, the extra free
    memory lets 917,504 fit without adapting, rope `auto` picks yarn:3.5 instead of yarn:3, and all four canaries
    change: that is the rope factor, not the body (`nvfp4-canary-yarn35.json`).
- **A5: green except one test that main fails the same way.** `cargo test --workspace`: 3,461 passed, 0 failed.
  The GPU profile (`a5-gpu-profile-*.log`):
  - `scripts/gpu-profile.ps1`: the leaf's stage (build and CTest, 138 of 138, 498.6 s) and the CPU oracle passed.
    Its cargo stage then died compiling on a full F: (os error 112), and was rerun with `-SkipKernelBuild` once the
    debug PDBs were deleted;
  - the serialized sweep stopped at its first red, core's `vision_load_gpu`. A vision load adds 328,239,040 bytes
    against the asserted 326,856,640 (1,382,400 more), and main 05ccde09 built and run alone gives the same two
    numbers, so this red is not #339's;
  - the packages it never reached (runtime, server, logging, vendor) ran with the same preflight, `IGNIS_GPU_PROFILE=1`
    and arguments plus `--no-fail-fast`: 153 test binaries green;
  - in all, 195 `#[ignore]`d tests passed and 1 failed. The five trellis tests are green: `trellis_projections_gpu`,
    `trellis_layers_gpu`, `trellis_logits_gpu`, `trellis_lanes_gpu`, `trellis_16g_gpu` (7.68 tokens per round).
    So are the NVFP4 guards: runtime's `decode_lanes_gpu` (the 8-lane greedy fixture recorded on main, every width),
    `vram_plan_gpu`, `cuda_leaf_dflash2_gpu`, `adaptation_gpu`, `openai_http_gpu`;
  - one SKIP line: `partial_exclusive_pool_study`, a CPU-only Flash-Next study (`#[ignore = "machine-local:
    IGNIS_FLASH_NEXT_TRACES, ~minutes of CPU"]`, untouched since #329) that returns without its traces. It is not a
    GPU test and does not ask `gpu_profile`.

## Part B (ticket 05b) - gates and simulation

Blocked by 05a. Everything below this line is Part B.

## Problem Statement

- M1 measured the candidate body weights-only in a PyTorch harness (KLD 0.0340 vs the served NVFP4's 0.0458 on the
  2026-09-24 windows).
- The ignis artifact adds:
  - our kernels' numerics (spec 02);
  - hq-e8-2b KV (+~0.005 at 4K in the 2026-09-24 finding, more at 32-64K);
  - the host embedding;
  - DFlash2 on a lossy target.
- The 5090 emulates a 16 GB card's memory with `--vram-budget-bytes`, but not its bandwidth or SM count.

## Solution

1. **Quality gates on the ignis artifact.** The owner's, 2026-10-21:

   | # | gate | how it is measured |
   |---|---|---|
   | 1 | **KLD ≤ 1.2×** the served NVFP4's | same suite, same engine path: `span_logits.rs` dumps to `Y:`, scored with `ref_kld.py`, serving configuration (hq-e8-2b, chunk 1024) |
   | 2 | **MMLU-Pro** (n=281) not significantly below BF16 (McNemar p > 0.05) | through the engine's logits on the Flash-Next study's chunks, if M2 step 0 clears the harness; otherwise on the corrected harness M2 names |
   | 3 | **DFlash2 tokens per round ≥ 90%** of the NVFP4 target's | the in-house teacher-forced replay (61,692 recorded rounds, branch `dflash2-sampled-acceptance-267`), same traces on both artifacts |
   | 4 | **loop rate** no worse than today's | the loop set (`.scratch/loops/`), same prompts and seeds on both artifacts |

   Plus *(proposed)* the KLD per domain of the Flash-Next held-out set, reported (Python is the known weak spot: ppl
   +8.6% in M1).
2. **Capacity simulation on the 5090.** An explicit budget of 17,179,869,184 bytes, minus 0, 1 and 3 GB of simulated
   desktop. For each: the plan lines, the adaptations taken (spec 04), the context and lanes reached, and a short
   serving run (load, prefill 128K, decode both lanes) that must succeed. The plan lines are checked against
   `PIANO-VRAM.md`'s estimates.
3. **Speed estimate, stated as such.**
   - Measured on the 5090: decode round time at 1/2 lanes and 8K/64K/256K, its bytes, and the kernels' achieved
     bandwidth.
   - The same with spec 02's SM caps (36 / 70 / 84 SMs).
   - Scaled to each 16 GB card's bandwidth by the study's model (`.scratch/27b-16gb-study-2026-10-20/scripts/budget.py`).
   - Published as an estimate with its assumptions.
4. **A user bench.** *(proposed)* `ignis-bench` gains a `small-card` preset that records:
   - card name, total VRAM, PCIe link;
   - plan lines and adaptations;
   - decode tok/s on the coding prompts and on agent-like turns;
   - TTFT at 8K/64K.

   It writes one JSON a user can paste into an issue. This is how phase 2 finds real numbers if no card is rented.

## Testing Decisions

- Gates 1-4 are run once, at the end of phase 1, on a free 5090 (the owner's practice: benchmarks are error detectors,
  run once at phase end). Each result goes into a finding.
- The simulation scenarios run as GPU tests where cheap: load and plan. The 128K serving run is a bench leg.

## Acceptance

1. Gate 1: KLD of the ignis 16 GB artifact on the 2026-09-24 suite, serving configuration, ≤ 1.2× the served NVFP4's
   on the same run's reference.
2. Gate 2: MMLU-Pro on the cleared or corrected harness, McNemar p > 0.05 against BF16.
3. Gate 3: DFlash2 tokens per round on the replay ≥ 90% of the NVFP4 artifact's.
4. Gate 4: loop rate on the loop set no worse than the NVFP4 artifact's (same prompts, seeds, guard settings).
5. The capacity simulation's three scenarios load, adapt as spec 04 says, serve the 128K leg, and their plans are
   within 2% of the estimates (or the estimates are corrected).
6. The speed estimate is published with its assumptions and the SM-cap measurements, labelled as not measured on a
   16 GB card.
7. The `small-card` bench preset exists and writes the JSON above.
8. One finding per gate group: quality, capacity, speed estimate, with `docs/findings/README.md` rows.

### As built: Part B (2026-10-24)

Main at 711b1d85 and the branches `340-replay` (gate 3) and `340-gates` (capacity, speed, `small-card`) on it, release
builds, on the RTX 5090 (PCIe Gen 3). Every trellis run loads as a 16 GB card would (owner,
2026-10-24): budget 17,179,869,184, `rtx50-16gb`, 262,144 context, 2 lanes, chunk 1,024, hq-e8-2b. The evidence is in
`.scratch/340/` (untracked); the findings are
[the quality gates](../../findings/2026-10-24-27b-16gb-quality-gates.md),
[the capacity on the 5090](../../findings/2026-10-24-27b-16gb-capacity-on-the-5090.md) and
[the speed estimate](../../findings/2026-10-24-27b-16gb-speed-estimate.md).

1. **Gate 1: pass.** Trellis KLD 0.04034 against the NVFP4's 0.05221 (chunk 1,024, the spec's shape) and 0.04916 (chunk
   2,048, as served): 0.7728x (95% CI [0.7368, 0.8144]) and 0.8207x (CI [0.7903, 0.8582]) against the 1.2x limit
   ([quality gates](../../findings/2026-10-24-27b-16gb-quality-gates.md), gate 1; `.scratch/340/quality/gate1-analysis.txt`).
   The NVFP4 baselines carry no YaRN, the NVFP4's best case.
2. **Gate 2: pass.** MMLU-Pro n = 281: trellis 65.84% against BF16's 67.97%, exact McNemar p = 0.3075 (pre-registered: pass
   iff p > 0.05) ([quality gates](../../findings/2026-10-24-27b-16gb-quality-gates.md), gate 2;
   `.scratch/340/quality/gate2-analysis.txt`). The per-domain KLD of set B is in the same finding.
3. **Gate 3: pass.** The #267 teacher-forced replay, ported onto main (`340-replay`), on its 200 recorded agent turns:
   DFlash2 commits 3.370 tokens a round on the trellis against 3.383 on the NVFP4, 0.996 (95% interval [0.993, 0.999]),
   traffic-weighted the same; the NVFP4 leg reproduces #267 (3.383 against 3.386) and the seams test holds forced rounds
   to live ones on both bodies. The NVFP4 leg runs the trellis's engine shape (262,144, rope none, chunk 1,024) at 8 lanes,
   not the served one ([quality gates](../../findings/2026-10-24-27b-16gb-quality-gates.md), gate 3;
   `.scratch/340/replay-run/handoff.md`).
4. **Gate 4: pass.** Loops 9/32 on the trellis against 14/32 on the NVFP4 (four prompts, seeds 101-108, loop guard at its
   default); one-sided Fisher exact p = 0.941 for "the trellis is worse" ([quality gates](../../findings/2026-10-24-27b-16gb-quality-gates.md),
   gate 4; `.scratch/340/loops/report.md`). Forced closes are
   15/32 against 8/32, reported and not part of the gate.
5. **Capacity: met at B0 and B1; at B3 the 128K leg does not fit, by the owner's decision.** The three scenarios load and
   adapt as spec 04 says, and their plans were measured on this build. B1 (16 GiB less 1 GiB) served the 131,072-token
   leg cold and decoded both lanes at once; B0's 128K leg is #339 A2's (an earlier build, not redone), with this build's
   two-lane decode on the speed cells. At B3 (16 GiB less 3 GiB) the load adapts the context to 65,536 on two lanes, and
   131,072 is 577 pages (36,928 tokens) short once `vram.adapt_spare_tokens` is held back; the leg ran at 65,472 prompt
   tokens and a 128K request is refused with a 400: the owner chose to run the leg at the adapted context and change no
   configuration, so AC 5's 128K leg is not met there. The plans are within 1% of the corrected estimates on every line (the
   study's workspace and pool formulas were stale: no hq prompt planes since ADR 0061, no context term)
   ([capacity](../../findings/2026-10-24-27b-16gb-capacity-on-the-5090.md); `.scratch/340/capacity/handoff.md`).
6. **Speed estimate: published.** The 5090 measured at 1 and 2 lanes, 8K / 64K / 253K, uncapped and under the 84 / 70 / 36
   SM caps, with bytes and achieved bandwidth, and the per-card estimate for the 5060 Ti 16 GB, 5070 Ti and 5080 with its
   rule and assumptions, titled and stated as not measured on a 16 GB card
   ([speed estimate](../../findings/2026-10-24-27b-16gb-speed-estimate.md); `.scratch/340/speed/handoff.md`).
7. **The `small-card` preset exists**: `ignis-bench small-card --endpoint <url> [--out <file>]` writes one JSON
   (`ignis-small-card/1`): the card (name, VRAM, PCIe link current and max, read after the decode), the plan lines and
   KV pages, the config values that shape the load with their sources, what the load adapted, decode tok/s on the
   coding prompts (thinking off) and three synthetic agent turns (thinking on: with it off they answer with a tool call,
   which reaches the client whole, so the first live run on main timed none of them), TTFT at 8K and 64K; never a key,
   a header or a path. The live runs found two things no route said, so `GET /v1/models` gained them: the running
   `decode_lanes` and the load's `adapted` steps (`""` when none). On merged main at B3 the record reads
   `running_decode_lanes` 2, `adapted` `max_context 262144 -> 65536 (adapt_context_step 65536, adapt_spare_tokens
   65536)`, decode 476 tok/s median on the coding prompts and 437 on the agent turns, 4.78 accepted drafts a round,
   and the 64K cell as not fitting (it names `--cells 65528`), with no note
   (`.scratch/340/final/live-b3-main-2.json`; the B0 run: `.scratch/340/small-card/live-b0.json`). The user guide's
   page: [measure a 16 GB card](https://github.com/xzzqqq/ignis-5080/blob/9259978846d7db1a0161253138f6e34006dfbb64/docs/user/guides/measure-a-16gb-card.md).
8. **One finding per gate group: done.** The three findings above, with their rows in
   [`docs/findings/README.md`](../../findings/README.md).

## Out of Scope

- A speed gate on a real 16 GB card: phase 2 (the owner deferred it, Q8).
- 12 GB: phase 2.

## Further Notes

- The M1 harness (`.scratch/27b-16gb-study-2026-10-20/m1/m1_exl3.py`) is the weights-only reference point. Its numbers
  are not the gate.
- KLD dumps are 2.03 GB per 4K window, so they go to `Y:`. F: has ~18 GB free.
