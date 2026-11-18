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
        let exploration=lab_trace::replay_with_check(&mut model,&zeros,&probe,2,3,
            lab_trace::ReplayCheck::Exploration).unwrap();
        assert_eq!(exploration["completed"],true);
        assert_eq!(exploration["executions_checked"],5,"exploration stopped on mode {mode}");
        assert_eq!(exploration["mismatches"],4);
        assert!(exploration["samples_ms"].as_array().unwrap().iter().all(|t|t.as_f64().unwrap()>0.0));
        let elements=probe["tokens"].as_u64().unwrap()*probe["weight"]["out"].as_u64().unwrap();
        let d=&exploration["diagnostics"];
        assert_eq!(d["total_elements"],elements*5);
        assert_eq!(d["actual_inf"],if mode==8 {4}else{0});
        assert_eq!(d["actual_nan"],0);
        assert_eq!(d["finite_elements"],elements*5-if mode==8 {4}else{0});
        assert_eq!(d["max_abs_error"],if mode==7 {1.0}else{0.0});
        if mode==7 {assert!(d["relative_l2"].is_null());}
        unsafe{kernel_lab::reload(&mut model,&mut pool,None)}.unwrap();
    }
    // Nonfinite references are counted explicitly and excluded from both norms.
    unsafe{kernel_lab::reload(&mut model,&mut pool,Some(&zero))}.unwrap();
    let mut synthetic=std::fs::read(&zeros).unwrap();
    let reference=(probe["x_bytes"].as_u64().unwrap()+probe["y_bytes"].as_u64().unwrap()) as usize;
    for (i,bits) in [0x3f80u16,0xc000,0x7fc0,0x7f80].iter().enumerate() {
        synthetic[reference+2*i..reference+2*i+2].copy_from_slice(&bits.to_le_bytes());
    }
    std::fs::write(&zeros,synthetic).unwrap();
    let result=lab_trace::replay_with_check(&mut model,&zeros,&probe,1,1,lab_trace::ReplayCheck::Exploration).unwrap();
    let d=&result["diagnostics"];
    assert_eq!(d["mismatches"],8);
    assert_eq!(d["reference_nan"],2);assert_eq!(d["reference_inf"],2);
    assert_eq!(d["actual_nan"],0);assert_eq!(d["actual_inf"],0);
    assert_eq!(d["max_abs_error"],2.0);assert_eq!(d["relative_l2"],1.0);
    unsafe{kernel_lab::reload(&mut model,&mut pool,None)}.unwrap();
    // Filtering happens before pinned allocation or copies for both graph and eager calls.
    let filtered=dir.join("filtered");std::fs::create_dir(&filtered).unwrap();
    assert!(lab_trace::start_filtered(&mut model,&mut pool,&filtered,1<<30,1<<30,selection,&[0]).is_err());
    assert_eq!(lab_trace::minimum_filtered(&model,selection,4,1,&[4]).unwrap(),0);
    lab_trace::start_filtered(&mut model,&mut pool,&filtered,1<<30,1<<30,selection,&[4]).unwrap();
    assert_eq!(observe(&model,&pool),expected);
    lab_trace::stop(&mut model,&mut pool).unwrap();
    let filtered_records:Vec<Value>=std::fs::read_to_string(filtered.join("records.jsonl")).unwrap().lines()
        .map(|l|serde_json::from_str(l).unwrap()).collect();
    assert!(!filtered_records.is_empty());
    assert!(filtered_records.iter().all(|r|r["tokens"]==4));
    assert!(filtered_records.iter().any(|r|r["weight"]["name"]=="layer.0.gate"));
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Capture C1 verify calls while the resident pool and captured engine support
/// two lanes. A held slot forces nonzero device slot/table selection.
#[test]
#[ignore = "exclusive RTX 5080; IGNIS_MODEL_ARTIFACT required"]
fn hq_attention_recording_restores_pages_residual_ring_and_masks() {
    let artifact_path=std::env::var("IGNIS_MODEL_ARTIFACT").unwrap();
    let dir=std::env::temp_dir().join(format!("ignis-lab-gqa-capture-{}",std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    let reader=Reader::open(Path::new(&artifact_path)).unwrap();
    let (plan,handles)=bind_model_scope_27b(&reader,Some(DraftModule::Dflash2)).unwrap();
    let mut device=CudaDevice::create(0).unwrap();
    let artifact=materialize(&reader,&plan,&mut device,None).unwrap();
    let spec=Speculation::new(SpeculativeBackend::Dflash2,7).unwrap();
    let mut model=load_qwen38_27b_with_decode_lanes(&reader,&artifact,&handles,128,1024,KvFormat::HqE8_2b,
        Some(spec),None,RopeScaling::NONE,PrefillAttention::default(),2).unwrap();
    let mut pool=SeqPool::create_with_speculation(&ModelConfig::qwen38_27b(),&SeqPoolBudget {
        kv_format:KvFormat::HqE8_2b,kv_page_group_count:32,max_context_tokens:1024,slot_count:2,
        retained_slot_count:0,retained_host_slot_count:0,
    },Some(spec.backend())).unwrap();
    assert_eq!(capture_decode_graphs(&model,&pool).unwrap().ready_mask,3);
    let observe_hq=|model:&Model,pool:&SeqPool|{
        let _held=pool.alloc(1024).unwrap();
        let mut all=vec![];
        for (length,remaining) in [(63usize,1),(64,3),(65,8),(620,8)]{
            let mut seq=pool.alloc(1024).unwrap();let mut logits=vec![0f32;248320];
            let prompt=vec![11;length];
            lab_trace::context(model,0,0,0).unwrap();
            prefill_program_sampled(model,pool,&mut seq,&prompt,0,SamplingParams::greedy(),Some(&mut logits)).unwrap();
            lab_trace::flush(model).unwrap();
            lab_trace::context(model,0,2,length as u64).unwrap();
            let lane=VerifyLane {remaining_tokens:remaining,..VerifyLane::greedy(&[])};
            let run=decode_program_verify_runs(model,pool,&mut [&mut seq],&[lane],7).unwrap().remove(0);
            lab_trace::flush(model).unwrap();
            all.push((logits.into_iter().map(f32::to_bits).collect::<Vec<_>>(),run.tokens));
        }
        all
    };
    let expected=observe_hq(&model,&pool);
    lab_trace::start_filtered(&mut model,&mut pool,&dir,1<<30,1<<30,"layer.3.attention",&[8]).unwrap();
    assert_eq!(observe_hq(&model,&pool),expected,"GQA instrumentation changed forward trajectory");
    lab_trace::stop(&mut model,&mut pool).unwrap();
    let records:Vec<Value>=std::fs::read_to_string(dir.join("records.jsonl")).unwrap().lines()
        .map(|l|serde_json::from_str(l).unwrap()).collect();
    assert_eq!(records.len(),4);
    for r in &records {
        assert_eq!(r["kind"],"gqa_attention");
        let result=lab_trace::replay_with_check(&mut model,&dir.join("data.bin"),r,2,3,lab_trace::ReplayCheck::Exploration).unwrap();
        assert_eq!(result["passed"],true,"GQA replay failed: {result}");
        assert_eq!(result["executions_checked"],5);
        assert_eq!(result["state_diagnostics"]["mismatched_bytes"],0);
        assert_eq!(result["state_diagnostics"]["checked_bytes"].as_u64().unwrap(),r["state_bytes"].as_u64().unwrap()*5);
    }
    // Optional hotreload coverage reuses exactly the same immutable HQ corpus.
    // The supplied default candidate must preserve builtin behavior. Fault
    // libraries use tests/fixtures/gqa_candidate_fault.cu with modes1/2/3.
    if let Ok(candidate)=std::env::var("IGNIS_GQA_CANDIDATE") {
        unsafe{kernel_lab::reload(&mut model,&mut pool,Some(Path::new(&candidate)))}.unwrap();
        let identity=lab_trace::gqa_status(&model);
        assert_eq!(identity["extension_loaded"],true);
        for r in &records {
            let result=lab_trace::replay_with_check(&mut model,&dir.join("data.bin"),r,1,2,lab_trace::ReplayCheck::Exploration).unwrap();
            assert_eq!(result["passed"],true,"default GQA candidate differs: {result}");
            assert_eq!(result["candidate_applied"],true);
            assert_eq!(result["implementation"],identity["implementation"]);
        }
        assert_eq!(observe_hq(&model,&pool),expected,"default candidate forward differs");
        if let Ok(faults)=std::env::var("IGNIS_GQA_FAULT_DIR") {
            for mode in 1..=3 {
                let fault=Path::new(&faults).join(format!("gqa-fault-{mode}.so"));
                assert!(unsafe{kernel_lab::reload(&mut model,&mut pool,Some(&fault))}.is_err());
                assert_eq!(lab_trace::gqa_status(&model),identity,"failed reload lost GQA generation");
                assert_eq!(lab_trace::replay(&mut model,&dir.join("data.bin"),&records[0],1,2).unwrap()["passed"],true);
            }
        }
        unsafe{kernel_lab::reload(&mut model,&mut pool,None)}.unwrap();
        assert_eq!(lab_trace::gqa_status(&model)["extension_loaded"],false);
    }
    let mut probe=records.last().unwrap().clone();
    let offset=probe["offset"].as_u64().unwrap() as usize;let bytes=probe["bytes"].as_u64().unwrap() as usize;
    let original=std::fs::read(dir.join("data.bin")).unwrap()[offset..offset+bytes].to_vec();
    probe["offset"]=json!(0);
    let modified=dir.join("modified.bin");
    // Damaged expected state reports exact-state differences without hiding
    // attention timing or conflating state bytes with BF16 output elements.
    let mut data=original.clone();let tail=data.len()-1;data[tail]^=1;
    std::fs::write(&modified,&data).unwrap();
    let result=lab_trace::replay_with_check(&mut model,&modified,&probe,1,2,lab_trace::ReplayCheck::Exploration).unwrap();
    assert_eq!(result["mismatches"],0);assert_eq!(result["state_diagnostics"]["mismatched_bytes"],3);
    assert_eq!(result["samples_ms"].as_array().unwrap().len(),2);
    // Invalid device metadata must be rejected on the host before a kernel
    // can dereference an out-of-range physical slot.
    let tensor_bytes=|tensor:&Value|tensor["shape"].as_array().unwrap().iter()
        .map(|n|n.as_u64().unwrap() as usize).product::<usize>()*match tensor["dtype"].as_u64().unwrap(){0=>2,1=>1,_=>4};
    let rows_offset=probe["tensors"].as_array().unwrap()[..6].iter().map(tensor_bytes).sum::<usize>();
    data=original.clone();data[rows_offset..rows_offset+4].copy_from_slice(&i32::MAX.to_le_bytes());
    std::fs::write(&modified,&data).unwrap();
    assert!(lab_trace::replay(&mut model,&modified,&probe,0,1).unwrap_err().contains("slot/mask"));
    data=original;data.pop();std::fs::write(&modified,&data).unwrap();
    assert!(lab_trace::replay(&mut model,&modified,&probe,0,1).unwrap_err().contains("truncated"));
    assert_eq!(observe_hq(&model,&pool),expected,"isolated replay modified resident sequence state");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[ignore = "exclusive RTX 5080; IGNIS_MODEL_ARTIFACT required; optional IGNIS_SEGMENT_CANDIDATE default policy"]
fn finite_segments_capture_replay_and_preserve_gdn_commit_liveouts() {
    let path=std::env::var("IGNIS_MODEL_ARTIFACT").unwrap();
    let kv_name=std::env::var("IGNIS_SEGMENT_TEST_KV_FORMAT").unwrap_or_else(|_|"hq-e8-2b".into());
    let kv=match kv_name.as_str(){"hq-e8-2b"=>KvFormat::HqE8_2b,"bf16"=>KvFormat::Bf16,_=>panic!("unsupported segment test KV format")};
    let hq=kv==KvFormat::HqE8_2b;
    let dir=std::env::temp_dir().join(format!("ignis-segment-capture-{}",std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    let reader=Reader::open(Path::new(&path)).unwrap();
    let (plan,handles)=bind_model_scope_27b(&reader,Some(DraftModule::Dflash2)).unwrap();
    let mut device=CudaDevice::create(0).unwrap();let artifact=materialize(&reader,&plan,&mut device,None).unwrap();
    let spec=Speculation::new(SpeculativeBackend::Dflash2,7).unwrap();
    let mut model=load_qwen38_27b_with_decode_lanes(&reader,&artifact,&handles,128,1024,kv,
        Some(spec),None,RopeScaling::NONE,PrefillAttention::default(),2).unwrap();
    let mut pool=SeqPool::create_with_speculation(&ModelConfig::qwen38_27b(),&SeqPoolBudget{
        kv_format:kv,kv_page_group_count:32,max_context_tokens:1024,slot_count:2,
        retained_slot_count:0,retained_host_slot_count:0},Some(spec.backend())).unwrap();
    capture_decode_graphs(&model,&pool).unwrap();
    let run=|model:&Model,pool:&SeqPool|{
        let _held=pool.alloc(128).unwrap();let mut seq=pool.alloc(1024).unwrap();
        let mut logits=vec![0f32;248320];lab_trace::context(model,0,0,0).unwrap();
        prefill_program_sampled(model,pool,&mut seq,&vec![11;620],0,SamplingParams::greedy(),Some(&mut logits)).unwrap();
        lab_trace::flush(model).unwrap();lab_trace::context(model,0,2,620).unwrap();
        let tokens=decode_program_verify_runs(model,pool,&mut [&mut seq],&[VerifyLane{remaining_tokens:3,..VerifyLane::greedy(&[])}],7)
            .unwrap().remove(0).tokens;
        lab_trace::flush(model).unwrap();(logits.into_iter().map(f32::to_bits).collect::<Vec<_>>(),tokens)
    };
    let expected=run(&model,&pool);
    lab_trace::start_filtered(&mut model,&mut pool,&dir,1<<30,1<<30,
        if hq {"layer.0.mlp_block,layer.3.gqa_block,layer.0.gdn_block"} else {"layer.0.mlp_block,layer.0.gdn_block"},&[8]).unwrap();
    assert_eq!(run(&model,&pool),expected,"segment capture changed real forward");
    lab_trace::stop(&mut model,&mut pool).unwrap();
    let records:Vec<Value>=std::fs::read_to_string(dir.join("records.jsonl")).unwrap().lines()
        .map(|s|serde_json::from_str(s).unwrap()).collect();
    assert_eq!(records.len(),if hq {3}else{2});
    for r in &records {
        assert_eq!(r["kind"],"segment");
        assert_eq!(r["reference"],"original_layer_chain");
        let result=lab_trace::replay_with_check(&mut model,&dir.join("data.bin"),r,2,3,lab_trace::ReplayCheck::Exploration).unwrap();
        assert_eq!(result["passed"],true,"segment replay: {result}");
        assert_eq!(result["executions_checked"],5);
        assert!(!result["outputs"].as_array().unwrap().is_empty());
    }
    let gdn=records.iter().find(|r|r["profile"]["kind"]==3).unwrap();
    for name in ["conv_record","key_record","value_record","gate_record"]{
        let binding=gdn["bindings"].as_array().unwrap().iter().find(|b|b["name"]==name).unwrap();
        assert_eq!(binding["flags"],6,"commit liveout lost mutable preimage/numerical role");
    }
    if let Ok(path)=std::env::var("IGNIS_SEGMENT_CANDIDATE"){
        unsafe{kernel_lab::reload(&mut model,&mut pool,Some(Path::new(&path)))}.unwrap();
        for r in &records{
            let result=lab_trace::replay_with_check(&mut model,&dir.join("data.bin"),r,1,2,lab_trace::ReplayCheck::Exploration).unwrap();
            assert_eq!(result["passed"],true,"default segment candidate: {result}");
            assert_eq!(result["candidate_applied"],true);
        }
        assert_eq!(run(&model,&pool),expected,"default segments changed forward");
        unsafe{kernel_lab::reload(&mut model,&mut pool,None)}.unwrap();
    }
    assert_eq!(run(&model,&pool),expected,"isolated segment replay changed resident state");
    if hq {
        // RoPE positions are analytic signed I32 values, independently of KV
        // positions. These deliberately modified inputs test that domain;
        // their output/state errors against the old reference are diagnostic.
        let original=std::fs::read(dir.join("data.bin")).unwrap();
        let mut probe=records.iter().find(|r|r["profile"]["kind"]==2).unwrap().clone();
        let offset=probe["offset"].as_u64().unwrap() as usize;let bytes=probe["bytes"].as_u64().unwrap() as usize;
        let pre_size=|b:&Value|{
            if b["present"]!=true || b["flags"].as_u64().unwrap()&3==0 {return 0usize;}
            b["shape"].as_array().unwrap().iter().map(|n|n.as_u64().unwrap() as usize).product::<usize>()*
                match b["dtype"].as_u64().unwrap(){0=>2,1=>1,_=>4}
        };
        let bindings=probe["bindings"].as_array().unwrap();
        let binding_offset=|name:&str|bindings.iter().take_while(|b|b["name"]!=name).map(pre_size).sum::<usize>();
        let (rope_offset,kv_offset)=(binding_offset("rope_positions"),binding_offset("positions"));
        let pristine=original[offset..offset+bytes].to_vec();probe["offset"]=json!(0);
        let modified=dir.join("signed-rope.bin");
        for start in [-4096i32,4096] {
            let mut data=pristine.clone();
            for j in 0..8usize {data[rope_offset+j*4..rope_offset+j*4+4].copy_from_slice(&(start+j as i32).to_le_bytes());}
            std::fs::write(&modified,&data).unwrap();
            let result=lab_trace::replay_with_check(&mut model,&modified,&probe,0,1,lab_trace::ReplayCheck::Exploration).unwrap();
            assert_eq!(result["completed"],true);assert_eq!(result["executions_checked"],1);
            assert_eq!(result["diagnostics"]["actual_nan"],0);assert_eq!(result["diagnostics"]["actual_inf"],0);
        }
        let mut invalid=pristine;invalid[kv_offset..kv_offset+4].copy_from_slice(&(-1i32).to_le_bytes());
        std::fs::write(&modified,&invalid).unwrap();
        assert!(lab_trace::replay(&mut model,&modified,&probe,0,1).unwrap_err().contains("positions"));
    }
    // Preserve malformed-file protection before invoking any recorded index.
    let mut invalid=gdn.clone();invalid["profile"]["tokens"]=json!(7);
    assert!(lab_trace::replay(&mut model,&dir.join("data.bin"),&invalid,0,1).is_err());
    std::fs::remove_dir_all(dir).unwrap();
}
