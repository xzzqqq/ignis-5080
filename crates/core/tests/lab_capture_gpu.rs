//! Real graph/eager capture and replay through the current model-local plugin.
//! IGNIS_TRELLIS_FAULT_DIR must contain fault-4.so, fault-7.so and fault-8.so,
//! built from tests/fixtures/trellis_candidate_fault.cu with matching TEST_MODE.
#![cfg(feature = "cuda")]
use std::path::Path;
use ignis_artifact::{bind_model_scope_27b, materialize, CudaDevice, DraftModule, Reader};
use ignis_core::compute::ModelConfig;
use ignis_core::{kernel_lab,lab_trace,KvFormat,PrefillAttention,RopeScaling,Speculation,SpeculativeBackend};
use ignis_core::model_load::{load_qwen38_27b_with_decode_lanes,Model};
use ignis_core::seq::{SeqPool,SeqPoolBudget};
use ignis_core::step::{capture_decode_graphs,prefill_program_sampled,decode_program_verify_runs,SamplingParams,VerifyLane};
use serde_json::{json,Value};

fn observe(model:&Model,pool:&SeqPool)->(Vec<u32>,Vec<i32>) {
    let mut seq=pool.alloc(128).unwrap();let mut logits=vec![0f32;248320];
    lab_trace::context(model,0,0,0).unwrap();
    prefill_program_sampled(model,pool,&mut seq,&[11,12,13,14],0,SamplingParams::greedy(),Some(&mut logits)).unwrap();
    lab_trace::flush(model).unwrap();
    lab_trace::context(model,0,2,4).unwrap();
    let run=decode_program_verify_runs(model,pool,&mut [&mut seq],&[VerifyLane::greedy(&[])],7).unwrap().remove(0);
    lab_trace::flush(model).unwrap();
    (logits.into_iter().map(f32::to_bits).collect(),run.tokens)
}

#[test]
#[ignore = "exclusive RTX 5080; IGNIS_MODEL_ARTIFACT and IGNIS_TRELLIS_FAULT_DIR required"]
fn recorded_calls_replay_with_padding_add_and_replaced_code() {
    let artifact_path=std::env::var("IGNIS_MODEL_ARTIFACT").unwrap();
    let fault_dir=std::env::var("IGNIS_TRELLIS_FAULT_DIR").unwrap();
    let dir=std::env::temp_dir().join(format!("ignis-lab-capture-{}",std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    let reader=Reader::open(Path::new(&artifact_path)).unwrap();
    let (plan,handles)=bind_model_scope_27b(&reader,Some(DraftModule::Dflash2)).unwrap();
    let mut device=CudaDevice::create(0).unwrap();
    let artifact=materialize(&reader,&plan,&mut device,None).unwrap();
    let spec=Speculation::new(SpeculativeBackend::Dflash2,7).unwrap();
    let mut model=load_qwen38_27b_with_decode_lanes(&reader,&artifact,&handles,128,128,KvFormat::Bf16,
        Some(spec),None,RopeScaling::NONE,PrefillAttention::default(),2).unwrap();
    let mut pool=SeqPool::create_with_speculation(&ModelConfig::qwen38_27b(),&SeqPoolBudget {
        kv_format:KvFormat::Bf16,kv_page_group_count:4,max_context_tokens:128,slot_count:2,
        retained_slot_count:0,retained_host_slot_count:0,
    },Some(spec.backend())).unwrap();
    assert_eq!(capture_decode_graphs(&model,&pool).unwrap().ready_mask,3);
    let expected=observe(&model,&pool);
    let selection="layer.0.gate,layer.0.down,layer.0.qkv,layer.0.z,layer.0.output,layer.3.q,layer.3.k,layer.3.v,layer.3.output,head";
    let minimum=lab_trace::minimum(&model,selection,4,1).unwrap();
    assert!(minimum>0);
    lab_trace::start(&mut model,&mut pool,&dir,1<<30,1<<30,selection).unwrap();
    assert_eq!(observe(&model,&pool),expected,"recording changed the forward pass");
    lab_trace::stop(&mut model,&mut pool).unwrap();
    assert_eq!(observe(&model,&pool),expected,"clean graphs changed the forward pass");
    let records:Vec<Value>=std::fs::read_to_string(dir.join("records.jsonl")).unwrap().lines()
        .map(|l|serde_json::from_str(l).unwrap()).collect();
    assert!(std::fs::metadata(dir.join("data.bin")).unwrap().len()>=minimum);
    for name in selection.split(',') {
        assert!(records.iter().any(|r|r["weight"]["name"]==name),"missing {name}");
    }
    assert!(records.iter().any(|r|r["phase"]==0 && r["tokens"]==4));
    assert!(records.iter().any(|r|r["phase"]==2 && r["tokens"]==8));
    for r in &records {
        let result=lab_trace::replay_with_check(&mut model,&dir.join("data.bin"),r,2,3,lab_trace::ReplayCheck::EveryExecution).unwrap();
        assert_eq!(result["passed"],true,"record {}: {result}",r["id"]);
        assert_eq!(result["executions_checked"],5);
        let mut padded=r.clone();padded["ldy"]=json!(r["ldy"].as_u64().unwrap()+128);
        assert_eq!(lab_trace::replay(&mut model,&dir.join("data.bin"),&padded,1,2).unwrap()["passed"],true,"padded output {}",r["id"]);
    }
    let mut invalid=records[0].clone();invalid["weight"]["trellis_offset"]=json!(u64::MAX);
    assert!(lab_trace::replay(&mut model,&dir.join("data.bin"),&invalid,1,1).is_err());
    let zero=Path::new(&fault_dir).join("fault-4.so");
    unsafe{kernel_lab::reload(&mut model,&mut pool,Some(&zero))}.unwrap();
    assert_ne!(observe(&model,&pool).0,expected.0,"full forward replayed stale code");
    let nonzero=records.iter().find(|r|r["weight"]["name"]=="layer.0.gate").unwrap();
    assert_eq!(lab_trace::replay(&mut model,&dir.join("data.bin"),nonzero,1,1).unwrap()["passed"],false,
        "local replay bypassed the candidate");
    unsafe{kernel_lab::reload(&mut model,&mut pool,None)}.unwrap();
    assert_eq!(observe(&model,&pool),expected);
    assert_eq!(lab_trace::replay(&mut model,&dir.join("data.bin"),nonzero,1,1).unwrap()["passed"],true);
    // Synthetic zero inputs and outputs isolate the delayed fault from model numerics.
    // Both fixtures return zero on launch 1, then a finite error / infinity on launch 2.
    let mut probe=records.iter().find(|r|r["weight"]["name"]=="layer.0.down").unwrap().clone();
    probe["offset"]=json!(0);
    let zeros=dir.join("zeros.bin");
    std::fs::write(&zeros,vec![0u8;probe["bytes"].as_u64().unwrap() as usize]).unwrap();
    for mode in [7,8] {
        let fault=Path::new(&fault_dir).join(format!("fault-{mode}.so"));
        unsafe{kernel_lab::reload(&mut model,&mut pool,Some(&fault))}.unwrap();
        let first=lab_trace::replay_with_check(&mut model,&zeros,&probe,2,3,
            lab_trace::ReplayCheck::FirstExecution).unwrap();
        assert_eq!(first["passed"],true,"first execution must match the reference");
        assert_eq!(first["executions_checked"],1);
        let every=lab_trace::replay_with_check(&mut model,&zeros,&probe,2,3,
            lab_trace::ReplayCheck::EveryExecution).unwrap();
        assert_eq!(every["passed"],false,"must catch delayed fault mode {mode}");
        assert_eq!(every["executions_checked"],2);
        assert_eq!(lab_trace::replay(&mut model,&zeros,&probe,2,3).unwrap()["passed"],false,
            "default regression API must check every execution");
        unsafe{kernel_lab::reload(&mut model,&mut pool,None)}.unwrap();
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
