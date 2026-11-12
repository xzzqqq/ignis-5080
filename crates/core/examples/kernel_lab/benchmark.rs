use super::*;
use ignis_core::step::{prefill_program_draw, decode_program_round, LaneDraw, Redirect};

/// C1 execution of prepared benchmark input. No intermediate values are replayed here.
pub fn run(model: &Model, pool: &SeqPool, settings: &Settings, case: &Value,
           after_step: &mut dyn FnMut(&Model, &str, u64) -> Result<()>) -> Result<Value> {
    let cap = integer(case, "output_tokens", 8192)?;
    if cap == 0 || cap >= settings.context { return Err("invalid output budget".into()); }
    let prompt = kernel_lab::tokens(&case["tokens"], (settings.context-cap) as usize, 248320)?;
    let stops: Vec<i32> = serde_json::from_value(case["stop_ids"].clone())?;
    if stops.is_empty() || stops.iter().any(|&x| !(0..248320).contains(&x)) { return Err("invalid stop ids".into()); }
    let close_id = integer(case,"think_close",248069)? as i32;
    if !(0..248320).contains(&close_id) {return Err("invalid reasoning close id".into());}
    let starts_in_reasoning=case["thinking_open"].as_bool().ok_or("missing thinking_open")?;
    let close=ignis_core::thinking_budget::ThinkingClose::new(vec![close_id as u32],close_id as u32)?;
    let mut thinking=ignis_core::thinking_budget::BudgetState::default();
    let mut seq = pool.alloc(settings.context)?;
    let mut logits = vec![0f32; 248320];
    let request_index=integer(case,"request_index",0)?;
    ignis_core::lab_trace::context(model,request_index,0,0)?;
    let start=Instant::now();
    prefill_program_draw(model,pool,&mut seq,&prompt,0,
        LaneDraw {sampling:SamplingParams::greedy(),permitted:&[],
            redirect:Redirect {stop_ids:&stops,close_id:if starts_in_reasoning && thinking.closed_at().is_none() {close_id} else {-1}}},Some(&mut logits))?;
    let prefill_ms=start.elapsed().as_secs_f64()*1000.0;
    after_step(model,"prefill",0)?;
    if logits.iter().any(|x|!x.is_finite()) { return Err("nonfinite benchmark logits".into()); }
    let mut tokens = vec![];
    let mut rounds = vec![];
    let mut finish = "length";
    let mut stop_token = None;
    while tokens.len() < cap as usize {
        let remaining = cap-tokens.len() as u32;
        let position = prompt.len()+tokens.len();
        ignis_core::lab_trace::context(model,request_index,if settings.spec.is_some(){2}else{1},position as u64)?;
        let start = Instant::now();
        let (emitted,extent) = if let Some(spec) = settings.spec {
            let mut runs = decode_program_verify_runs(model,pool,&mut [&mut seq],&[VerifyLane {
                remaining_tokens:remaining,stop_ids:&stops,reasoning_close:if starts_in_reasoning && thinking.closed_at().is_none() {close_id} else {-1},
                ..VerifyLane::greedy(&[])
            }],spec.draft_tokens())?;
            let run = runs.remove(0);
            (run.tokens,run.extent)
        } else {
            let mut draw = LaneDraw::free(SamplingParams::greedy());
            draw.redirect = Redirect {stop_ids:&stops,close_id:if starts_in_reasoning && thinking.closed_at().is_none() {close_id} else {-1}};
            let result = decode_program_round(model,pool,&mut [&mut seq],&[draw],None)?;
            (result.into_iter().map(|r|r.token).collect::<Vec<_>>(),0)
        };
        let elapsed = start.elapsed().as_secs_f64()*1000.0;
        after_step(model,if settings.spec.is_some() {"verify"} else {"decode"},position as u64)?;
        if emitted.is_empty() || emitted.len()>remaining as usize { return Err("invalid decode progress".into()); }
        rounds.push(json!({"ms":elapsed,"emitted":emitted.len(),"extent":extent,
            "position":position,"graph_launches":program_stats(model,pool)?.graph_launches}));
        for token in emitted {
            if stops.contains(&token) { finish="stop";stop_token=Some(token);break; }
            thinking.commit(&close,tokens.len() as u32,token as u32);
            tokens.push(token);
        }
        if stop_token.is_some() {break;}
    }
    let decode_ms: f64 = rounds.iter().map(|r|r["ms"].as_f64().unwrap()).sum();
    Ok(json!({"tier":"forward","id":case["id"],"input_tokens":prompt.len(),
        "prefill_ms":prefill_ms,"decode_ms":decode_ms,"rounds":rounds,
        "tokens":tokens,"output_tokens":tokens.len(),"finish_reason":finish,"stop_token":stop_token,
        "thinking_open_at_end":starts_in_reasoning && thinking.closed_at().is_none(),"logits_sha256":hash(&logits.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>()),
        "timing_scope":"synchronous engine calls; capture/IO excluded; no HTTP or scheduler"}))
}
