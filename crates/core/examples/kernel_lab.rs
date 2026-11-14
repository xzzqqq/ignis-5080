//! Persistent, serial JSONL experiment engine. See docs/user/contributing/kernel-lab.md.
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::time::Instant;
use ignis_artifact::{bind_model_scope_27b, materialize, CudaDevice, DraftModule, FrontendSet, Reader};
use ignis_core::compute::ModelConfig;
use ignis_core::kernel_lab;
use ignis_core::model_load::{load_qwen38_27b_with_decode_lanes, Model};
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::step::{capture_decode_graphs, decode_program_verify_runs, program_stats, SamplingParams, VerifyLane};
use ignis_core::{KvFormat, PrefillAttention, RopeScaling, Speculation, SpeculativeBackend};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[path = "kernel_lab/benchmark.rs"]
mod benchmark;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn emit(v: Value) { println!("{v}"); io::stdout().flush().unwrap(); }
fn integer(v: &Value, name: &str, default: u32) -> Result<u32> {
    match v.get(name) { None => Ok(default), Some(x) => Ok(u32::try_from(x.as_u64().ok_or("expected unsigned integer")?)?) }
}
fn hash(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }

struct Settings { context: u32, chunk: u32, lanes: u32, spec: Option<Speculation> }

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
    let mut recording = false;
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
                "capture_minimum" => Ok(json!({"prefill_bytes_lower_bound":ignis_core::lab_trace::minimum(&model,
                    v["names"].as_str().unwrap_or(""),v["tokens"].as_u64().ok_or("missing tokens")?,
                    v["requests"].as_u64().ok_or("missing requests")?)?})),
                "capture_start" => {
                    let path=Path::new(v["path"].as_str().ok_or("missing capture path")?);
                    ignis_core::lab_trace::start(&mut model,&mut pool,path,
                        v["pin_bytes"].as_u64().ok_or("missing pin budget")?,v["disk_bytes"].as_u64().ok_or("missing disk budget")?,v["names"].as_str().unwrap_or(""))?;
                    recording = true;
                    Ok(json!({"recording":true}))
                },
                "capture_stop" => {ignis_core::lab_trace::stop(&mut model,&mut pool)?;recording=false;Ok(json!({"recording":false}))},
                "replay_batch" => {
                    let data=Path::new(v["data"].as_str().ok_or("missing data path")?);
                    let records=v["records"].as_array().ok_or("missing records")?;
                    let mut rows=Vec::new();
                    for record in records {
                        let row=ignis_core::lab_trace::replay(&mut model,data,record,
                            integer(&v,"warmup",0)?,integer(&v,"samples",1)?)?;
                        let passed=row["passed"]==true;
                        rows.push(row);
                        if !passed {break;}
                    }
                    Ok(json!(rows))
                },
                "replay" => Ok(ignis_core::lab_trace::replay(&mut model,Path::new(v["data"].as_str().ok_or("missing data path")?),
                    &v["record"],integer(&v,"warmup",1)?,integer(&v,"samples",3)?)?),
                "benchmark" => {
                    let mut result = benchmark::run(&model, &pool, &settings, &v["case"], &mut |m,_,_| {
                        ignis_core::lab_trace::flush(m)?;Ok(())
                    })?;
                    result["recording"] = json!(recording);
                    if recording {result["timing_scope"] = json!("instrumented recording; not performance");}
                    Ok(result)
                },
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
                if matches!(v["op"].as_str(), Some("replay_batch" | "benchmark" | "capture_start" | "capture_stop" | "replay")) { return Err(e); }
            },
        }
    }
    Ok(())
}
