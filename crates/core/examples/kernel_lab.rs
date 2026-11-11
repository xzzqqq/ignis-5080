//! Persistent, serial JSONL experiment engine. See docs/user/contributing/kernel-lab.md.
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::time::Instant;
use ignis_artifact::{bind_model_scope_27b, materialize, CudaDevice, Device, DraftModule, FrontendSet, Reader};
use ignis_core::compute::ModelConfig;
use ignis_core::kernel_lab;
use ignis_core::model_load::{load_qwen38_27b_with_decode_lanes, Model};
use ignis_core::seq::{Seq, SeqPool, SeqPoolBudget};
use ignis_core::step::{capture_decode_graphs, decode_program_batch_sampled, decode_program_verify_runs,
    prefill_program_sampled, program_stats, SamplingParams, VerifyLane};
use ignis_core::{KvFormat, PrefillAttention, RopeScaling, Speculation, SpeculativeBackend};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn emit(v: Value) { println!("{v}"); io::stdout().flush().unwrap(); }
fn integer(v: &Value, name: &str, default: u32) -> Result<u32> {
    match v.get(name) { None => Ok(default), Some(x) => Ok(u32::try_from(x.as_u64().ok_or("expected unsigned integer")?)?) }
}
fn hash(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }

struct Settings { context: u32, chunk: u32, lanes: u32, spec: Option<Speculation> }

fn measure(model: &Model, pool: &SeqPool, settings: &Settings, v: &Value) -> Result<Value> {
    let output_tokens = integer(v, "output_tokens", 32)?;
    if output_tokens == 0 || output_tokens >= settings.context { return Err("invalid output_tokens".into()); }
    let prompts = v["prompts"].as_array().ok_or("prompts must be token arrays")?;
    if prompts.is_empty() || prompts.len() > settings.lanes as usize { return Err("invalid lane count".into()); }
    let prompts: Vec<Vec<i32>> = prompts.iter().map(|p| kernel_lab::tokens(p,
        (settings.context - output_tokens) as usize, 248320)).collect::<std::result::Result<_, _>>()?;
    let mut seqs: Vec<Seq<'_>> = prompts.iter().map(|_| pool.alloc(settings.context)).collect::<std::result::Result<_, _>>()?;
    let mut logits = vec![vec![0f32; 248320]; prompts.len()];
    let began = Instant::now();
    for ((seq, prompt), logit) in seqs.iter_mut().zip(&prompts).zip(&mut logits) {
        prefill_program_sampled(model, pool, seq, prompt, 0, SamplingParams::greedy(), Some(logit))?;
    }
    let prefill_ms = began.elapsed().as_secs_f64() * 1000.0;
    if logits.iter().flatten().any(|x| !x.is_finite()) { return Err("nonfinite logits".into()); }
    let mut outputs: Vec<Vec<i32>> = vec![vec![]; prompts.len()];
    let mut round_ms = vec![];
    let mut widths = vec![];
    while outputs.iter().any(|o| o.len() < output_tokens as usize) {
        let active: Vec<usize> = (0..outputs.len()).filter(|&i| outputs[i].len() < output_tokens as usize).collect();
        let mut all: Vec<Option<&mut Seq<'_>>> = seqs.iter_mut().map(Some).collect();
        let mut refs: Vec<&mut Seq<'_>> = active.iter().map(|&i| all[i].take().unwrap()).collect();
        let start = Instant::now();
        let runs = if let Some(spec) = settings.spec {
            let lanes: Vec<_> = active.iter().map(|&i| VerifyLane {
                remaining_tokens: output_tokens - outputs[i].len() as u32, ..VerifyLane::greedy(&[])
            }).collect();
            decode_program_verify_runs(model, pool, &mut refs, &lanes, spec.draft_tokens())?.into_iter().map(|r| r.tokens).collect::<Vec<_>>()
        } else {
            decode_program_batch_sampled(model, pool, &mut refs, &vec![SamplingParams::greedy(); active.len()])?
                .into_iter().map(|t| vec![t]).collect()
        };
        round_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        widths.push(active.len());
        if program_stats(model, pool)?.graph_launches != 1 { return Err("round did not use a captured graph".into()); }
        for (&i, run) in active.iter().zip(runs) {
            if run.is_empty() { return Err("empty decode round".into()); }
            outputs[i].extend(run);
        }
    }
    let logits_sha256: Vec<_> = logits.iter().map(|l| hash(&l.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())).collect();
    Ok(json!({"tier":"forward", "input_tokens":prompts.iter().map(Vec::len).collect::<Vec<_>>(),
        "prefill_ms":prefill_ms, "decode_ms":round_ms.iter().sum::<f64>(), "round_ms":round_ms,
        "widths":widths, "tokens":outputs, "logits_sha256":logits_sha256,
        "eos_policy":"fixed token budget; not the service benchmark"}))
}

fn layer(model: &Model, pool: &SeqPool, device: &mut CudaDevice, settings: &Settings, v: &Value) -> Result<Value> {
    let index = integer(v, "layer", 0)?;
    let count = integer(v, "tokens", 1)?;
    if index >= 64 || count == 0 || count > settings.chunk { return Err("invalid layer or token width".into()); }
    let bytes = std::fs::read(v["input"].as_str().ok_or("layer input must be a BF16 file")?)?;
    if bytes.len() != count as usize * 5120 * 2 { return Err("layer input size mismatch".into()); }
    let input = device.allocate(bytes.len() as u64)?;
    let output = device.allocate(bytes.len() as u64)?;
    let result = (|| -> Result<Value> {
        device.copy_h2d(&input, 0, &bytes)?;
        device.synchronize()?;
        let seq = pool.alloc(settings.context)?;
        let began = Instant::now();
        if index % 4 == 3 {
            ignis_core::gqa_layer::run_gqa_layer(model, pool, &seq, index, &input, &output, u64::from(count))?;
        } else {
            ignis_core::gdn_layer::run_gdn_layer(model, pool, &seq, index, &input, &output, u64::from(count))?;
        }
        let elapsed = began.elapsed().as_secs_f64() * 1000.0;
        let mut out = vec![0; bytes.len()];
        device.copy_d2h(&output, 0, &mut out)?;
        device.synchronize()?;
        if out.chunks_exact(2).any(|b| !f32::from_bits(u32::from(u16::from_le_bytes([b[0],b[1]])) << 16).is_finite()) {
            return Err("nonfinite layer output".into());
        }
        if let Some(path) = v["output"].as_str() { std::fs::write(path, &out)?; }
        Ok(json!({"tier":"layer", "layer":index, "tokens":count, "wall_ms":elapsed,
            "input_sha256":hash(&bytes), "output_sha256":hash(&out), "state":"fresh sequence, position zero"}))
    })();
    device.deallocate(input)?;
    device.deallocate(output)?;
    result
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 { return Err("usage: kernel_lab ARTIFACT SETTINGS_JSON; commands on stdin".into()); }
    let config: Value = serde_json::from_str(&std::fs::read_to_string(&args[2])?)?;
    let spec = match config["speculation"].as_str().unwrap_or("off") {
        "off" => None, "dflash2" => Some(Speculation::new(SpeculativeBackend::Dflash2, 7)?),
        _ => return Err("speculation must be off or dflash2".into())
    };
    let settings = Settings { context:integer(&config,"max_context",4096)?, chunk:integer(&config,"prefill_chunk",128)?,
        lanes:integer(&config,"lanes",2)?, spec };
    if settings.context == 0 || settings.context > 131072 || settings.chunk == 0 || settings.chunk % 128 != 0 ||
        settings.chunk > settings.context || !(1..=2).contains(&settings.lanes) { return Err("invalid fixed-model settings".into()); }
    let kv = KvFormat::parse(config["kv_format"].as_str().unwrap_or("bf16"))?;
    let began = Instant::now();
    let reader = Reader::open(Path::new(&args[1]))?;
    let frontend = FrontendSet::from_reader(&reader)?;
    let (plan, handles) = bind_model_scope_27b(&reader, spec.map(|_| DraftModule::Dflash2))?;
    let open_s = began.elapsed().as_secs_f64();
    let phase = Instant::now();
    let mut device = CudaDevice::create(0)?;
    let context_s = phase.elapsed().as_secs_f64();
    let phase = Instant::now();
    let _artifact = materialize(&reader, &plan, &mut device, None)?;
    let materialize_s = phase.elapsed().as_secs_f64();
    let phase = Instant::now();
    let mut model = load_qwen38_27b_with_decode_lanes(&reader, &_artifact, &handles, settings.chunk,
        settings.context, kv, spec, None, RopeScaling::NONE, PrefillAttention::default(), settings.lanes)?;
    let model_s = phase.elapsed().as_secs_f64();
    let phase = Instant::now();
    let mut pool = SeqPool::create_with_speculation(&ModelConfig::qwen38_27b(), &SeqPoolBudget {
        kv_format:kv, kv_page_group_count:settings.context.div_ceil(64) * settings.lanes,
        max_context_tokens:settings.context, slot_count:settings.lanes, retained_slot_count:0, retained_host_slot_count:0,
    }, spec.map(|s|s.backend()))?;
    let pool_s = phase.elapsed().as_secs_f64();
    let phase = Instant::now();
    let capture = capture_decode_graphs(&model, &pool)?;
    if capture.ready_mask != (1 << settings.lanes) - 1 { return Err("initial graph capture incomplete".into()); }
    let capture_s = phase.elapsed().as_secs_f64();
    let mut generation = 0u64;
    emit(json!({"event":"ready", "pid":std::process::id(), "generation":generation, "settings":config,
        "startup_s":began.elapsed().as_secs_f64(), "phases_s":{"open_bind_frontend":open_s,"cuda_context":context_s,
        "weight_materialize":materialize_s,"model_workspace":model_s,"pool":pool_s,"graph_capture":capture_s}}));
    for line in io::stdin().lock().lines() {
        let start = Instant::now();
        let v: Value = match serde_json::from_str(&line?) { Ok(v) => v, Err(e) => { emit(json!({"ok":false,"error":e.to_string()})); continue; } };
        let result = (|| -> Result<Value> {
            match v["op"].as_str().ok_or("missing op")? {
                "reload" => {
                    let path = v["path"].as_str().ok_or("path must be builtin or an absolute library path")?;
                    let (candidate, sha) = if path == "builtin" { (None, None) } else {
                        if !Path::new(path).is_absolute() { return Err("candidate path must be absolute".into()); }
                        let bytes = std::fs::read(path)?;
                        let sha = hash(&bytes);
                        if v["sha256"].as_str() != Some(&sha) { return Err("candidate sha256 mismatch".into()); }
                        (Some(Path::new(path)), Some(sha))
                    };
                    unsafe { kernel_lab::reload(&mut model, &mut pool, candidate)?; }
                    generation += 1;
                    Ok(json!({"candidate":path,"sha256":sha}))
                },
                "measure" => measure(&model, &pool, &settings, &v),
                "layer" => layer(&model, &pool, &mut device, &settings, &v),
                "tokenize" => Ok(json!({"tokens":frontend.tokenizer().encode(v["text"].as_str().ok_or("missing text")?)?})),
                _ => Err("unknown op".into()),
            }
        })();
        match result {
            Ok(value) => emit(json!({"ok":true,"id":v["id"],"generation":generation,"wall_s":start.elapsed().as_secs_f64(),"result":value})),
            Err(e) => {
                emit(json!({"ok":false,"id":v["id"],"generation":generation,"wall_s":start.elapsed().as_secs_f64(),"error":e.to_string()}));
                // Execution failures may leave a poisoned CUDA context. The
                // campaign must restart instead of timing later candidates.
                if matches!(v["op"].as_str(), Some("measure" | "layer")) { return Err(e); }
            },
        }
    }
    Ok(())
}
