//! A model's mutable state image, derived from its topology
//! (`ModelConfig::state_image`, spec flash-next/05), agrees with what the
//! leaf's sequence pool lays out for one slot (`ignis_seq_pool_plan`).
//!
//! The Rust derivation is what a load prints and what a host plan charges;
//! the leaf's planner is what a slot really holds. A section either one
//! sized with the other model's number -- the 27B's 16 attention layers in
//! Flash-Next's residual window, say -- turns this red.
//!
//! Host-only: planning a pool makes no CUDA call. `#[ignore]`d because the
//! `cuda` feature it needs is only built for the GPU profile's run, like
//! `kv_format_leaf_agreement.rs`.

#![cfg(feature = "cuda")]

use ignis_core::compute::ModelConfig;
use ignis_core::kv_format::KvFormat;
use ignis_core::seq::{SeqPool, SeqPoolBudget};

const PAGES: u32 = 8;
const SLOTS: u32 = 3;

fn budget(kv_format: KvFormat, retained_host_slot_count: u32) -> SeqPoolBudget {
    SeqPoolBudget {
        kv_format,
        kv_page_group_count: PAGES,
        max_context_tokens: 512,
        slot_count: SLOTS,
        retained_slot_count: 0,
        retained_host_slot_count,
    }
}

/// The 27B's image is exactly one host retained slot as the leaf lays it
/// out: every section from a 256-byte boundary.
#[test]
#[ignore]
fn the_27b_image_is_the_slot_the_leaf_lays_out() {
    let cfg = ModelConfig::qwen38_27b();
    for format in [KvFormat::Bf16, KvFormat::HqE8_2b] {
        let plan = SeqPool::plan(&cfg, &budget(format, 1), None).unwrap_or_else(|e| panic!("27B {format}: {e}"));
        assert_eq!(cfg.state_image(format).slot_bytes(), plan.retained_host_bytes, "27B {format}");
    }
}
