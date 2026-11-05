# 02 — YaRN on Qwen3.8-Flash-Next

**Decision:** [ADR 0059](../../adr/0059-flash-next-applies-yarn-on-q-and-k.md) · **Branch:**
`fn-yarn` · **Builds on:** [01 — YaRN RoPE scaling as a load option](01-yarn.md)

## Why

Flash-Next is trained over 262,144 positions and served at 589,824 (RTX 5090) and 1,048,576
(RTX PRO 6000). Spec 01 gave the 27B the YaRN table; Flash-Next's load refused one, its options
crossed none, and its attention refused an attention factor, so every Flash-Next position past
262,144 ran the linear table extrapolated. The model card's recipe for that range is static
YaRN, and HF, vLLM and SGLang all run it the same way on this model (ADR 0059's Context).

## Seam

The table, built once per load, and the two places Flash-Next's kernels rotate.

- **Rust.** `EngineOptions::rope_scaling` (default `RopeScaling::NONE`).
  `flash_next_options(.., rope_scaling)` crosses its four scalars exactly as the 27B's
  `load_options` does; `plan_flash_next_reservations` and `load_flash_next` take it in one
  `FlashNextLoadShape` (`EngineOptions::load_shape`), beside the other values both are asked
  under.
  `FlashNextEngine`, `FlashNextLeaf` and the server's Flash-Next plan and load pass the one
  value the load was configured with (`EngineShape::rope_scaling`), and the server's KV-disk
  identity takes the leaf's own.
- **Leaf load** (`flash_next::bind_model`). It takes the `ignis::RopeScaling` that
  `ignis_model_load` read off the four scalars and has already refused by name if unusable
  (`ignis::rope_scaling_rejection`); `flash_next::text_rope` builds the linear table without
  scaling (`rope_linear_frequencies`, untouched) and `ignis::rope_yarn_frequencies(
  topology.rope_theta, topology.rotary_dim, kOriginalPositions, scaling)` with it, the choice
  the 27B's table makes too (`ignis::rope_frequencies`). `kOriginalPositions` is Flash-Next's
  own 262,144, never the 27B's constant.
- **Kernels.** `indexer::Rope` gains `scale` (1.0f by default); `rope_from` copies the table's
  `attention_factor` into it. `prepare_kernel` (`qsa.cu`: 24 q heads and 2 k heads) and
  `rope_row` (`indexer.cu`: queries, and pooled block keys at the block's first position)
  round `cosf(phi) * scale` and `sinf(phi) * scale` to BF16, where they rounded `cosf(phi)` and
  `sinf(phi)`, both through one device helper (`indexer::rope_pair`, `rope.cuh`).
  `fn_qsa_attention` no longer refuses an attention factor.
- **MTP.** Nothing of its own: its attention layer and its indexer read the trunk's
  `Context::rope` (`views_of`, `program.cu`; `mtp::layer`, `mtp.cu`).

Nothing else moves: the angle is still `fp32(position) * inv_freq` in FP32 (ADR 0059 D2), and
the 27B's table, op and convention are untouched.

## Acceptance

1. `flash_next_options` crosses the four scalars of a YaRN scaling and, for `none`, the
   factor 0 (the leaf's linear branch) with `none`'s own three ramp scalars, as the 27B's
   `load_options` does: the bits the KV-disk identity records. *(cargo,
   `model_load.rs`)*
2. The server's Flash-Next load builds its engine options from the shape's rope scaling,
   plans with those options' value and opens KV-disk under the leaf's own: plan, load and
   identity cannot disagree. *(cargo, `crates/server/src/runtime.rs`)*
3. `rope_from(rope_yarn_frequencies(1e7, 64, 262144, f))` matches an independent FP64
   evaluation of HF's `_compute_yarn_parameters` within 2 FP32 ulps on every pair, for f in
   {2, 2.25, 3, 4}. *(CTest, host)*
4. The ramp bounds read off that table are 14 and 22: pairs 13 and 14 are the linear
   frequency, 15 is not, 22 and 31 are the linear one over f. *(CTest, host)*
5. `scale == float(0.1 ln f + 1)` for each f, and the load's own builder
   (`flash_next::text_rope`) gives that table for a YaRN scaling. *(CTest, host)*
6. Without scaling (factor 0 or 1) the leaf's table is bit for bit the one before this change
   (recorded literals), with `scale == 1.0f`. *(CTest, host)*
7. At factor 4, `prepare_kernel`'s and `rope_row`'s outputs equal a host restatement
   (`round_bf16(fp32(cos(fp32(pos) * inv)) * m)` and the BF16 rotation) at positions 0,
   262,143, 262,144, 589,823 and 1,048,575 (block keys at the block-aligned 0, 262,140,
   262,144, 589,820 and 1,048,572), either side of a BF16 rounding boundary accepted where it
   falls within CUDA's 2-ulp bound on `cosf`/`sinf`; fewer than a tenth of their rotary values
   are the linear table's restatement. *(CTest, GPU)*
8. At factor 4 the indexer's selected blocks equal a host top-k (the documented tie rule)
   over host-rotated queries and keys, and at least one row selects other blocks under the
   linear table. *(CTest, GPU)*
9. A `none` load's logits are byte-identical to a capture taken on the same card before this
   change: 2,560 real-text tokens in 1,024-token chunks, so the last chunk selects
   sparsely. *(GPU, `flash_next_rope_logits_gpu.rs`, `logits-capture` feature)*
10. A `yarn:4` load serves: every logit is finite, and every row of the same prompt but the
    first differs from the `none` capture. The first stays the capture's: it attends to itself
    alone, and a softmax over one key is 1 whatever the table makes of its score. *(GPU, same
    test)*
11. The existing Flash-Next CTests and GPU tests pass unchanged on a `none` load. **Met
    2026-10-19** on the RTX 5090 (driver 610.88), with the review fixes' kernel, under
    `IGNIS_GPU_PROFILE=1`: the 22 `flash_next|rope` CTests, `flash_next_forward_gpu`,
    `flash_next_pinned_gpu`, `flash_next_counters_gpu`, `flash_next_residual_tap_gpu`,
    `flash_next_reuse_gpu`, `flash_next_serving_gpu` and `flash_next_acceptance_gpu` (G1
    102/102, every KLD under its limit, MMLU-Pro 70.82% against the accepted floor miss), and
    this spec's logits test against its committed capture. `flash_next_speculative_gpu` and
    `flash_next_reload_gpu` last ran on 2e352391, before the rope sites shared a helper; that
    refactor moved no bit (the logits capture 0 of 2,560 rows, forward G1 102/102). One serving
    test, `the_default_plan_is_offloaded_at_524288_tokens_and_the_floor_opt_in_warns`, first missed
    its margin (an expert cache 2 GiB past the floor) by 110 MB with the desktop's VRAM on the
    card, and passed alone once the derived budget read 27.24 GB instead of 26.43 GB: its plan
    lines besides the cache are the bytes they were on 2026-10-08, and it missed the same way
    on 2026-10-10.
12. The KV-disk spec records that reusing files across restarts must bump
    `FORMAT_VERSION`: files written before this change label linear-rotated pages `yarn:4`.

## Out of scope (follow-ups)

- **The long-context legs:** a needle and a multi-hop retrieval at ~512K on the RTX 5090
  (589,824, `auto` giving 2.25) and at 1,048,576 on the RTX PRO 6000 (factor 4, MTP on), each
  beside a `none` control that must degrade past 262,144. A later agent runs them on the merged
  result; Flash-Next's prefill there also needs the request timeout raised past 900 s.
  The ~512K leg ran on 2026-10-19 (hq-e8-2b, speculation off, one question per cell):
  `auto` 5/5 needles and 1/2 chains, `none` 4/5 and the same 1/2, so the `none` control did
  not degrade past 262,144 on this probe. The 1M leg ran the same day on the RTX PRO 6000
  (1,048,576, hq-e8-2b, MTP 3 on, one question per cell): `auto` (`yarn:4`) 4/5 needles and
  0/2 chains, `none` 5/5 and 1/2. The control did not degrade there either, and `auto`'s extra
  misses are the deepest facts (948.9K, 970.6K), each answered with another fact's number. MTP
  ran on the scaled table with no error. Neither leg separates the tables: one seed, one filler
  ([finding](https://github.com/xzzqqq/ignis-5080/blob/9259978846d7db1a0161253138f6e34006dfbb64/docs/findings/2026-10-19-flash-next-yarn-in-the-reference-engines.md)).
- **hq-e8-2b under `m`-scaled K** is measured by those legs at both KV formats, not here.
- **`original_max_position_embeddings` on the topology.** A named constant until a
  checkpoint differs.
