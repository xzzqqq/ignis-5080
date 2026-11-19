//! Dump the engine's BF16 logits for every position of a set of token
//! windows (the KLD against the BF16 checkpoint, 2026-09-24).
//!
//! ```text
//! cargo run --release -p ignis-core --features cuda --example span_logits -- \
//!     ARTIFACT KV_FORMAT WINDOWS_JSON OUT_DIR [PREFILL_CHUNK [TAIL [TAIL_STEP]]]
//! ```
//!
//! `WINDOWS_JSON` is a JSON array of token-id arrays. Each window is
//! prefilled on a fresh sequence through the serving route (1024-token
//! chunks unless `PREFILL_CHUNK` says otherwise -- the chunk width picks the
//! GEMM routes, so it moves the distribution; engine default policy) and its `[len][vocab]` little-endian BF16
//! logits land in `OUT_DIR/wNNN.bin`. Row `i` is the distribution of the
//! token after `window[i]`. With `TAIL`, only the last `TAIL` rows are read
//! (a long window's head is prefilled first, without logits, so a 32K window
//! costs 1 GB of host memory rather than 16). Without `TAIL_STEP`, `TAIL`
//! must be a multiple of the chunk; original whole-window chunk cuts also
//! require the head prefix length to be chunk-aligned. With `TAIL_STEP`,
//! only the tail is submitted in smaller calls (e.g. TAIL32/TAIL_STEP8);
//! the head retains the configured prefill chunk. These are eager prefill
//! traversals, not verify-graph replay. The output head's internal 32-column
//! readout blocks do not change the backbone's attention width.
//! `IGNIS_MODEL_EXCEPTION_WEIGHTS=fp8`, the server's own variable for
//! `--model-exception-weights`, holds the BF16 exception tensors in FP8
//! (GitHub #341); unset, they are held as stored, as the server's default.

use std::io::Write;
use std::path::{Path, PathBuf};

use ignis_artifact::{CudaDevice, ModelScope, VisionResidency, Reader, bind_model_scope_27b_with, materialize};
use ignis_core::{ExceptionWeights, KvFormat};
use ignis_core::compute::ModelConfig;
use ignis_core::model_load::load_qwen38_27b;
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::step::{prefill_program, prefill_program_span_logits};

const SERVING_PREFILL_CHUNK: u32 = 1024;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if !(5..=8).contains(&args.len()) {
        eprintln!("usage: span_logits ARTIFACT KV_FORMAT WINDOWS_JSON OUT_DIR [PREFILL_CHUNK [TAIL [TAIL_STEP]]]");
        std::process::exit(2);
    }
    let prefill_chunk: u32 = args
        .get(5)
        .map_or(SERVING_PREFILL_CHUNK, |v| v.parse().unwrap_or_else(|e| panic!("prefill chunk: {e}")));
    let tail: Option<usize> = args.get(6).map(|v| v.parse().unwrap_or_else(|e| panic!("tail: {e}")));
    let tail_step: Option<usize> = args.get(7).map(|v| v.parse().unwrap_or_else(|e| panic!("tail step: {e}")));
    if let Some(step) = tail_step {
        let tail = tail.expect("TAIL_STEP requires TAIL");
        assert!(step > 0 && step <= prefill_chunk as usize && tail > 0 && tail % step == 0,
                "TAIL_STEP must be positive, <= PREFILL_CHUNK, and divide the positive TAIL");
    } else if let Some(tail) = tail {
        assert!(tail > 0 && tail % prefill_chunk as usize == 0, "TAIL {tail} is not a positive multiple of the chunk {prefill_chunk}");
    }
    let kv_format = KvFormat::parse(&args[2]).unwrap_or_else(|e| panic!("kv format: {e}"));
    let windows: Vec<Vec<i32>> = serde_json::from_str(
        &std::fs::read_to_string(&args[3]).unwrap_or_else(|e| panic!("read {}: {e}", args[3])),
    )
    .unwrap_or_else(|e| panic!("parse windows: {e}"));
    let out_dir = PathBuf::from(&args[4]);
    std::fs::create_dir_all(&out_dir).unwrap_or_else(|e| panic!("create {}: {e}", out_dir.display()));
    let longest = windows.iter().map(Vec::len).max().unwrap_or(0) as u32;
    let max_context = longest.div_ceil(64) * 64 + 64;

    let reader = Reader::open(Path::new(&args[1])).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let exceptions = std::env::var("IGNIS_MODEL_EXCEPTION_WEIGHTS")
        .map_or(Ok(ExceptionWeights::default()), |v| ExceptionWeights::parse(&v))
        .unwrap_or_else(|e| panic!("IGNIS_MODEL_EXCEPTION_WEIGHTS: {e}"));
    eprintln!("exception weights: {exceptions}");
    let scope = ModelScope { draft: None, vision: false, vision_residency: VisionResidency::Device, fp8_exceptions: exceptions.is_fp8() };
    let (plan, handles) = bind_model_scope_27b_with(&reader, scope).unwrap_or_else(|e| panic!("bind: {e}"));
    let mut device = CudaDevice::create(0).unwrap_or_else(|e| panic!("CUDA: {e}"));
    let artifact = materialize(&reader, &plan, &mut device, None).unwrap_or_else(|e| panic!("materialize: {e}"));
    let model = load_qwen38_27b(&reader, &artifact, &handles, prefill_chunk, max_context, kv_format)
        .unwrap_or_else(|e| panic!("model load: {e}"));
    let pool = SeqPool::create(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format,
            kv_page_group_count: max_context.div_ceil(64),
            max_context_tokens: max_context,
            slot_count: 1,
            retained_slot_count: 0,
            retained_host_slot_count: 0,
        },
    )
    .unwrap_or_else(|e| panic!("pool: {e}"));
    let vocab = ModelConfig::qwen38_27b().vocab as usize;

    for (index, window) in windows.iter().enumerate() {
        let began = std::time::Instant::now();
        if tail_step.is_some() {
            assert!(tail.unwrap() <= window.len(), "window {index} is shorter than TAIL");
        }
        let head = tail.map_or(0, |tail| window.len().saturating_sub(tail));
        let mut rows = vec![0u16; (window.len() - head) * vocab];
        {
            let mut sequence = pool.alloc(max_context).unwrap_or_else(|e| panic!("alloc: {e}"));
            if head > 0 {
                prefill_program(&model, &pool, &mut sequence, &window[..head], 0, None)
                    .unwrap_or_else(|e| panic!("window {index} head: {e}"));
            }
            let step = tail_step.unwrap_or(window.len() - head);
            assert!(step > 0, "window {index} has no tail tokens");
            for offset in (0..window.len() - head).step_by(step) {
                let end = (offset + step).min(window.len() - head);
                prefill_program_span_logits(
                    &model, &pool, &mut sequence, &window[head + offset..head + end],
                    (head + offset) as u64, &mut rows[offset * vocab..end * vocab],
                ).unwrap_or_else(|e| panic!("window {index}, tail offset {offset}: {e}"));
            }
        }
        let path = out_dir.join(format!("w{index:03}.bin"));
        let mut file = std::io::BufWriter::new(
            std::fs::File::create(&path).unwrap_or_else(|e| panic!("create {}: {e}", path.display())),
        );
        let bytes: Vec<u8> = rows.iter().flat_map(|v| v.to_le_bytes()).collect();
        file.write_all(&bytes).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
        println!("window {index}: {} tokens in {:.2}s -> {}", window.len(), began.elapsed().as_secs_f64(), path.display());
    }
}
