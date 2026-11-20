//! The verify round's width-cost curve (GitHub #267): wall time of one
//! DFlash2 round at draft windows 1..7 (verify widths 2..8 columns per lane),
//! and of a plain decode round (width 1), at 1, 2, 4 and 8 lanes up to the
//! load's, each lane standing at the same context.
//!
//! ```text
//! cargo run --release -p ignis-runtime --features cuda --example verify_width_cost -- \
//!     [--lanes N] [--prefill-chunk N] [--kv-format FORMAT] [--max-context N] ARTIFACT TOKENS_BIN CONTEXTS ROUNDS OUT_JSONL [WINDOWS]
//! ```
//!
//! `TOKENS_BIN` is int32 LE text (any real stream). `CONTEXTS` is an ascending
//! comma list: lane `i` reads the text from `i * last context`, so the lanes
//! are distinct, and is prefilled to the first context, measured, extended by
//! prefill to the next, measured again. Per load (plain, then DFlash2-k for
//! each k in `WINDOWS`, default 1..7), each at the artifact's body's shape
//! (`support/replay_shape.rs`: the trellis one in `rtx50-16gb`'s 16 GiB plan
//! at its two lanes, the NVFP4 one at eight or `--lanes`) planned with the
//! server's plan pieces (`support/replay_load.rs`), graphs captured: per
//! context and lane count `n`, `ROUNDS` greedy rounds
//! over the first `n` lanes after 3 warm-up rounds, each timed on the host
//! around the call (the call returns after its device work). One JSON line
//! per (load, context, lanes): median, mean and p10/p90 round milliseconds.
//! Greedy, because a round's cost does not depend on what it accepts: every
//! lane's `k + 1` columns are computed whatever the accept keeps.

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use ignis_artifact::{body_27b, Reader};
use ignis_core::model_load::Model;
use ignis_core::seq::{Seq, SeqPool};
use ignis_core::step::{decode_program_batch_sampled, decode_program_verify_runs, prefill_program_sampled, SamplingParams, VerifyLane};

#[path = "support/replay_shape.rs"]
mod replay_shape;
#[path = "support/replay_load.rs"]
mod replay_load;

use replay_shape::{LoadShape, ShapeFlags};

const WARMUP: usize = 3;
const LANE_COUNTS: [usize; 4] = [1, 2, 4, 8];

fn stats(mut samples: Vec<f64>) -> serde_json::Value {
    samples.sort_by(f64::total_cmp);
    let at = |q: f64| samples[((samples.len() - 1) as f64 * q).round() as usize];
    serde_json::json!({
        "median_ms": at(0.5), "mean_ms": samples.iter().sum::<f64>() / samples.len() as f64,
        "p10_ms": at(0.1), "p90_ms": at(0.9), "samples": samples.len(),
    })
}

/// Lane `i`'s text: `stride` tokens from `i * stride`.
fn lane_text(text: &[i32], lane: usize, stride: usize) -> &[i32] {
    &text[lane * stride..(lane + 1) * stride]
}

/// Prefill every lane's text from `from` to `to` (its first `from` tokens
/// already prefilled, plus whatever the timed rounds committed after them).
fn extend_lanes(model: &Model, pool: &SeqPool, seqs: &mut [Seq<'_>], text: &[i32], stride: usize, from: usize, to: usize) {
    for (i, seq) in seqs.iter_mut().enumerate() {
        let position = seq.stats().position;
        prefill_program_sampled(model, pool, seq, &lane_text(text, i, stride)[from..to], position, SamplingParams::greedy(), None)
            .unwrap_or_else(|e| panic!("prefill: {e}"));
    }
}

fn main() {
    let mut args: Vec<String> = std::env::args().collect();
    let flags = ShapeFlags::take(&mut args).unwrap_or_else(|e| panic!("{e}"));
    if !(6..=7).contains(&args.len()) {
        eprintln!(
            "usage: verify_width_cost [--lanes N] [--prefill-chunk N] [--kv-format FORMAT] [--max-context N] ARTIFACT TOKENS_BIN CONTEXTS ROUNDS OUT_JSONL [WINDOWS]"
        );
        std::process::exit(2);
    }
    let bytes = std::fs::read(&args[2]).unwrap_or_else(|e| panic!("read {}: {e}", args[2]));
    let text: Vec<i32> = bytes.chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let contexts: Vec<usize> = args[3].split(',').map(|c| c.parse().expect("context")).collect();
    assert!(contexts.windows(2).all(|w| w[0] < w[1]), "contexts must ascend");
    let stride = *contexts.last().unwrap();
    let rounds: usize = args[4].parse().expect("rounds");
    let windows: Vec<u32> = args
        .get(6)
        .map_or_else(|| (1..=7).collect(), |v| v.split(',').map(|w| w.parse().expect("window")).collect());
    let mut out = std::fs::OpenOptions::new().create(true).append(true).open(&args[5]).expect("open out");

    let reader = Reader::open(Path::new(&args[1])).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let base = LoadShape::new(body_27b(&reader), flags).unwrap_or_else(|e| panic!("load shape: {e}"));
    let lanes = base.lanes as usize;
    let lane_counts: Vec<usize> = LANE_COUNTS.into_iter().filter(|&n| n <= lanes).collect();
    assert!(text.len() >= lanes * stride, "the text holds {} tokens, not {}", text.len(), lanes * stride);
    // Room for the prompt plus every timed round's full commit at every context.
    let committed_max = contexts.len() * lane_counts.len() * (WARMUP + rounds) * 8;
    let lane_context = ((stride + committed_max + 64) as u32).div_ceil(64) * 64;
    assert!(lane_context <= base.max_context, "a lane's {lane_context} tokens outgrow the load's {}", base.max_context);
    let mut record = |label: &str, window: u32, context: usize, lanes: usize, samples: Vec<f64>| {
        let line = serde_json::json!({
            "load": label, "kv_format": base.kv_format.as_str(), "max_context": base.max_context, "window": window, "verify_width": window + 1, "lanes": lanes,
            "context": context, "round": stats(samples),
        });
        println!("{line}");
        writeln!(out, "{line}").unwrap();
    };

    // The plain decode round (one column per lane), then each window.
    for window in std::iter::once(0).chain(windows.iter().copied()) {
        let loaded = replay_load::load(&reader, LoadShape { window, ..base })
            .unwrap_or_else(|e| panic!("load (window {window}): {e}"));
        {
            let (model, pool) = (&loaded.model, &loaded.pool);
            let mut seqs: Vec<Seq<'_>> =
                (0..lanes).map(|_| pool.alloc(lane_context).unwrap_or_else(|e| panic!("alloc: {e}"))).collect();
            let label = if window == 0 { "plain".to_string() } else { format!("dflash2-{window}") };
            let mut prefilled = 0;
            for &context in &contexts {
                extend_lanes(model, pool, &mut seqs, &text, stride, prefilled, context);
                prefilled = context;
                for &lanes in &lane_counts {
                    let mut samples = Vec::with_capacity(rounds);
                    for index in 0..WARMUP + rounds {
                        let mut refs: Vec<&mut Seq<'_>> = seqs.iter_mut().take(lanes).collect();
                        let began = Instant::now();
                        if window == 0 {
                            decode_program_batch_sampled(model, pool, &mut refs, &vec![SamplingParams::greedy(); lanes])
                                .unwrap_or_else(|e| panic!("decode: {e}"));
                        } else {
                            decode_program_verify_runs(model, pool, &mut refs, &vec![VerifyLane::greedy(&[]); lanes], window)
                                .unwrap_or_else(|e| panic!("round: {e}"));
                        }
                        if index >= WARMUP {
                            samples.push(began.elapsed().as_secs_f64() * 1e3);
                        }
                    }
                    record(&label, window, context, lanes, samples);
                }
            }
        }
        loaded.release();
    }
}
