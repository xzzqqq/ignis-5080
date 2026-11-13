//! Code lifetime, graph invalidation and rollback against a real resident model.
#![cfg(feature = "cuda")]
use std::path::Path;
use ignis_artifact::{bind_model_scope_27b, materialize, model_scope_27b_of, CudaDevice, DraftModule, ModelScope, Reader};
use ignis_core::compute::ModelConfig;
use ignis_core::kernel_lab::reload;
use ignis_core::model_load::{load_qwen38_27b_with_decode_lanes, Model};
use ignis_core::seq::{Seq, SeqPool, SeqPoolBudget};
use ignis_core::step::{capture_decode_graphs, decode_program_batch_sampled, decode_program_verify_runs,
    prefill_program_sampled, program_stats, SamplingParams, VerifyLane};
use ignis_core::{KvFormat, PrefillAttention, RopeScaling, Speculation, SpeculativeBackend};

fn observation(model: &Model, pool: &SeqPool, spec: bool) -> (Vec<u32>, Vec<Vec<i32>>) {
    let mut seqs: Vec<Seq<'_>> = (0..2).map(|_| pool.alloc(128).unwrap()).collect();
    let mut logits = vec![0f32; 248320];
    for (i, seq) in seqs.iter_mut().enumerate() {
        let prompt = [11 + i as i32, 12, 13, 14];
        prefill_program_sampled(model, pool, seq, &prompt, 0, SamplingParams::greedy(), Some(&mut logits)).unwrap();
    }
    assert!(logits.iter().all(|x| x.is_finite()));
    let mut outputs = vec![vec![], vec![]];
    for _ in 0..3 {
        let mut refs = seqs.iter_mut().collect::<Vec<_>>();
        let runs = if spec {
            decode_program_verify_runs(model, pool, &mut refs, &[VerifyLane::greedy(&[]), VerifyLane::greedy(&[])], 7)
                .unwrap().into_iter().map(|r| r.tokens).collect::<Vec<_>>()
        } else {
            decode_program_batch_sampled(model, pool, &mut refs, &[SamplingParams::greedy(); 2])
                .unwrap().into_iter().map(|t| vec![t]).collect()
        };
        assert_eq!(program_stats(model, pool).unwrap().graph_launches, 1);
        for (out, run) in outputs.iter_mut().zip(runs) { out.extend(run); }
    }
    (logits.into_iter().map(f32::to_bits).collect(), outputs)
}

#[test]
#[ignore = "requires an exclusive GPU, artifact, real candidate and six fault plugins; see kernel-lab guide"]
fn reload_recaptures_and_refusals_preserve_the_previous_generation() {
    let path = std::env::var("IGNIS_TRELLIS_ARTIFACT").expect("IGNIS_TRELLIS_ARTIFACT");
    let candidate = std::env::var("IGNIS_TRELLIS_CANDIDATE").expect("IGNIS_TRELLIS_CANDIDATE");
    let faults = std::env::var("IGNIS_TRELLIS_FAULT_DIR").expect("IGNIS_TRELLIS_FAULT_DIR");
    let reader = Reader::open(Path::new(&path)).unwrap();
    let (plan, handles) = bind_model_scope_27b(&reader, Some(DraftModule::Dflash2)).unwrap();
    let text_len = model_scope_27b_of(&reader, ModelScope::default()).unwrap().len();
    let mut device = CudaDevice::create(0).unwrap();
    let artifact = materialize(&reader, &plan, &mut device, None).unwrap();
    for spec in [None, Some(Speculation::new(SpeculativeBackend::Dflash2, 7).unwrap())] {
        let selected = if spec.is_some() { &handles[..] } else { &handles[..text_len] };
        let mut model = load_qwen38_27b_with_decode_lanes(&reader, &artifact, selected, 128, 128,
            KvFormat::Bf16, spec, None, RopeScaling::NONE, PrefillAttention::default(), 2).unwrap();
        let mut pool = SeqPool::create_with_speculation(&ModelConfig::qwen38_27b(), &SeqPoolBudget {
            kv_format:KvFormat::Bf16, kv_page_group_count:4, max_context_tokens:128, slot_count:2,
            retained_slot_count:0, retained_host_slot_count:0,
        }, spec.map(|s|s.backend())).unwrap();
        assert_eq!(capture_decode_graphs(&model,&pool).unwrap().ready_mask, 3);
        let memory = model.stats().vram_bytes;
        let baseline = observation(&model, &pool, spec.is_some());
        // The pool outlives this deliberately lifetime-erased handle. This tests
        // the C guard in addition to the Rust borrow check on the normal API.
        let active = unsafe { pool.alloc(128).unwrap().into_static() };
        assert!(unsafe { reload(&mut model, &mut pool, None) }.unwrap_err().contains("empty"));
        drop(active);
        for mode in [1, 2, 3, 5, 6] {
            let path = Path::new(&faults).join(format!("fault-{mode}.so"));
            assert!(unsafe { reload(&mut model, &mut pool, Some(&path)) }.is_err(), "fault {mode}");
            assert_eq!(observation(&model, &pool, spec.is_some()), baseline, "rollback {mode}");
        }
        for _ in 0..3 {
            unsafe { reload(&mut model, &mut pool, Some(Path::new(&candidate))) }.unwrap();
            assert_eq!(observation(&model, &pool, spec.is_some()), baseline, "real plugin");
            assert_eq!(model.stats().vram_bytes, memory);
            let rejected = Path::new(&faults).join("fault-3.so");
            assert!(unsafe { reload(&mut model, &mut pool, Some(&rejected)) }.is_err());
            assert_eq!(observation(&model, &pool, spec.is_some()), baseline, "previous plugin retained");
            // A graph-safe wrong kernel changes the actual result. If old graphs
            // were reused, same-code parity alone would never catch that bug.
            let zero = Path::new(&faults).join("fault-4.so");
            unsafe { reload(&mut model, &mut pool, Some(&zero)) }.unwrap();
            let changed = observation(&model, &pool, spec.is_some());
            assert_ne!(changed.0, baseline.0, "candidate did not execute");
            assert!(changed.1.iter().flatten().all(|&t| t == 0), "stale decode/verify graph: {:?}", changed.1);
            unsafe { reload(&mut model, &mut pool, None) }.unwrap();
            assert_eq!(observation(&model, &pool, spec.is_some()), baseline, "restored builtin");
        }
    }
}
