//! Three frozen SWE-bench request snapshots at C1. No tool execution.
//! Uses server phase timings, never SSE arrival times or chunk counts.
use std::path::Path;
use std::time::Instant;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::client::HttpEndpoint;
use crate::small_card;
use crate::time::{unix_now, utc_timestamp};
use crate::ttft;

pub const MANIFEST: &str = include_str!("../fixtures/coding_agent/manifest.json");
pub const FIXTURES: [&str; 3] = [
    include_str!("../fixtures/coding_agent/astropy__astropy-12907.json"),
    include_str!("../fixtures/coding_agent/django__django-10097.json"),
    include_str!("../fixtures/coding_agent/sympy__sympy-11618.json"),
];
pub const OUTPUT_CAP: u32 = 8192;
pub const SAMPLES: usize = 3;
pub const CAPACITY_TOKENS: usize = 131_072;

pub fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Protocol overrides only. The frozen conversation and tools stay intact.
pub fn request(mut body: Value, capacity: bool) -> Value {
    let fields = json!({
        "stream": false, "ignis_timing": true, "class": "agent",
        "enable_thinking": !capacity, "reasoning_effort": if capacity { "none" } else { "max" },
        "preserve_thinking": true,
        "temperature": 0, "seed": 0, "top_p": 1, "top_k": 0,
        "presence_penalty": 0, "frequency_penalty": 0,
        "max_tokens": if capacity { 64 } else { OUTPUT_CAP }, "ignore_eos": false,
    });
    body.as_object_mut().unwrap().extend(fields.as_object().unwrap().clone());
    body
}

fn write(path: &Path, value: &Value) -> Result<(), String> {
    let data = serde_json::to_string_pretty(value).map_err(|e| e.to_string())? + "\n";
    std::fs::write(path, data).map_err(|e| format!("{}: {e}", path.display()))
}

fn effective<'a>(config: &'a Value, family: &str, group: &str, key: &str) -> &'a Value {
    config[group][family].get(key).unwrap_or(&config[group][key])
}

/// A missing or inconsistent count invalidates a measurement; failures stay
/// in the record and prevent its three-sample median from being published.
pub fn sample(response: &Value, expected: u64, total_ms: f64) -> Value {
    let usage = &response["usage"];
    let timing = &usage["ignis_timing"];
    let input = usage["prompt_tokens"].as_u64();
    let output = usage["completion_tokens"].as_u64();
    let cached = usage["prompt_tokens_details"]["cached_tokens"].as_u64();
    let decode_ms = timing["decode_ms"].as_f64();
    let finish = response["choices"][0]["finish_reason"].as_str();
    let message = &response["choices"][0]["message"];
    let empty = ["content", "reasoning_content"].iter().all(|k| message[k].as_str().unwrap_or("").is_empty())
        && message["tool_calls"].as_array().is_none_or(|a| a.is_empty());
    let mut problems = Vec::new();
    if input != Some(expected) { problems.push("prompt token count differs from tokenize"); }
    if cached != Some(0) { problems.push("prefix cache is not proven cold"); }
    if timing["computed_prefill_tokens"].as_u64() != Some(expected) || timing["prefill_passes"] != 1 {
        problems.push("prefill did not compute exactly one full prompt");
    }
    if output.is_none_or(|n| n == 0) || timing["output_tokens"].as_u64() != output {
        problems.push("missing, zero or inconsistent output token count");
    }
    if decode_ms.is_none_or(|t| !t.is_finite() || t <= 0.0) { problems.push("missing or invalid decode time"); }
    if timing["prefill_ms"].as_f64().is_none_or(|t| !t.is_finite() || t < 0.0)
        || timing["elapsed_ms"].as_f64().is_none_or(|t| !t.is_finite() || t < decode_ms.unwrap_or(0.0))
    { problems.push("missing or invalid server phase times"); }
    if !matches!(finish, Some("stop" | "tool_calls" | "length")) { problems.push("missing or failed finish reason"); }
    if empty { problems.push("empty visible output (reasoning, text and tools)"); }
    let valid = problems.is_empty();
    json!({
        "valid": valid, "problems": problems, "empty_output": empty,
        "truncated": finish == Some("length"), "finish_reason": finish,
        "input_tokens": input, "output_tokens": output, "cached_tokens": cached,
        "prefill_ms": timing["prefill_ms"], "decode_ms": decode_ms,
        "server_elapsed_ms": timing["elapsed_ms"], "request_total_ms": total_ms,
        "decode_tokens_per_s": if valid { Some(output.unwrap() as f64 * 1000.0 / decode_ms.unwrap()) } else { None },
        "speculative": timing["speculative"],
    })
}

pub fn median(samples: &[Value]) -> Option<f64> {
    if samples.len() != SAMPLES || samples.iter().any(|s| s["valid"] != true) { return None; }
    ttft::median(&samples.iter().filter_map(|s| s["decode_tokens_per_s"].as_f64()).collect::<Vec<_>>())
}

fn measure(api: &HttpEndpoint, body: &Value, expected: u64, out: &Path, name: &str) -> Result<Value, String> {
    eprintln!("coding-agent: {name} ({expected} input tokens)");
    let start = Instant::now();
    let result = api.post_json("/v1/chat/completions", body);
    let total_ms = start.elapsed().as_secs_f64() * 1000.0;
    let (raw, mut row) = match result {
        Ok(response) => {
            let row = sample(&response, expected, total_ms);
            (response, row)
        }
        Err(error) => (json!({"error": error}), json!({"valid": false, "error": error, "request_total_ms": total_ms})),
    };
    let file = format!("{name}.response.json");
    write(&out.join(&file), &raw)?;
    row["response_file"] = json!(file);
    eprintln!("coding-agent: {name}: valid={}, output={}, decode tok/s={}", row["valid"], row["output_tokens"], row["decode_tokens_per_s"]);
    Ok(row)
}

fn count(api: &HttpEndpoint, body: &Value) -> Result<u64, String> {
    let value = api.post_json("/v1/tokenize", body)?;
    value["count"].as_u64().ok_or_else(|| "tokenize returned no count".into())
}

fn git(args: &[&str]) -> Option<String> {
    let output = std::process::Command::new("git").args(args).output().ok()?;
    output.status.success().then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// A new output directory is required: a rerun never overwrites evidence.
/// Capacity is a separate invocation and separate record, never a speed cell.
pub fn run(endpoint: &str, out: &Path, capacity: bool) -> Result<(), String> {
    if let Some(parent) = out.parent() { std::fs::create_dir_all(parent).map_err(|e| e.to_string())?; }
    std::fs::create_dir(out).map_err(|e| format!("{} (must be new): {e}", out.display()))?;
    let result = run_in_dir(endpoint, out, capacity);
    if let Err(error) = &result {
        let path = out.join("run.json");
        let mut report = std::fs::read(&path).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok())
            .unwrap_or_else(|| json!({"schema": "ignis-coding-agent/1"}));
        report["passed"] = json!(false);
        report["error"] = json!(error);
        write(&path, &report)?;
    }
    result
}

fn run_in_dir(endpoint: &str, out: &Path, capacity: bool) -> Result<(), String> {
    let api = HttpEndpoint::new(endpoint.trim_end_matches('/'));
    let config = api.get_json("/v1/config")?;
    let origins = api.get_json("/v1/config/origins")?;
    let models = api.get_json("/v1/models")?;
    let catalog = api.get_json("/v1/models/catalog")?;
    let family = small_card::loaded_family(&catalog).ok_or("catalog names no loaded family")?;
    let reuse = effective(&config, &family, "reuse", "prompt");
    if reuse != false { return Err("restart with --reuse-prompt false (fixed fixtures must stay cold)".into()); }
    if effective(&config, &family, "model", "loop_guard") != "off" { return Err("model.loop_guard must be off".into()); }
    let mut view = serde_json::to_value(small_card::config_view(&config, Some(&origins), &family)).unwrap();
    for (group, keys) in [
        ("reuse", &["prompt", "retained_device", "retained_host"][..]),
        ("server", &["request_timeout", "stream_timeout", "timeout_reply"][..]),
        ("model", &["thinking_budget", "idle_unload"][..]),
        ("vision", &["enabled", "max_tokens", "residency"][..]),
    ] {
        for key in keys { view[format!("{group}.{key}")] = effective(&config, &family, group, key).clone(); }
    }
    let mut report = json!({
        "schema": "ignis-coding-agent/1", "kind": if capacity { "capacity" } else { "decode" },
        "passed": false, "finished_at": null,
        "started_at": utc_timestamp(unix_now()), "bench_version": env!("CARGO_PKG_VERSION"),
        "code_commit": git(&["rev-parse", "HEAD"]), "working_tree": git(&["status", "--porcelain"]),
        "endpoint": small_card::redact_url(endpoint), "config": view, "models": models,
        "family": family, "profile": small_card::profile_name(&origins),
        "concurrency": 1, "prefix_cache": "disabled; each usage must report cached_tokens=0",
        "protocol": request(json!({}), capacity),
        "client_timeout_seconds": api.request_timeout().map(|d| d.as_secs()),
        "timing": "committed output tokens / model-thread wall time from final prefill chunk to scheduler Done; includes first decode round and completion scheduling, independent of HTTP buffering; first anchor sampled in prefill, committed in decode",
        "fixtures": serde_json::from_str::<Value>(MANIFEST).unwrap(), "requests": [],
    });
    let metrics = small_card::metrics_url_from_config(endpoint, Some(&config)).ok().map(HttpEndpoint::new);
    let save_metrics = |name: &str| -> Option<String> {
        let text = metrics.as_ref()?.get_text("/metrics").ok()?;
        std::fs::write(out.join(name), text).ok()?;
        Some(name.into())
    };
    report["metrics_before"] = json!(save_metrics("metrics-before.prom"));
    write(&out.join("run.json"), &report)?;
    let mut passed = true;
    if capacity {
        let template = small_card::ServerTemplate::new(endpoint);
        let prompts = ttft::generate_cell_prompts_with_suffix(
            &template, CAPACITY_TOKENS, 1, "\nReply with just OK.",
        )?;
        let body = request(json!({"messages": [{"role": "user", "content": prompts.prompts[0]}]}), true);
        write(&out.join("capacity.request.json"), &body)?;
        let expected = count(&api, &body)?;
        if expected != CAPACITY_TOKENS as u64 { return Err(format!("capacity template counted {expected}, expected {CAPACITY_TOKENS}")); }
        let row = measure(&api, &body, expected, out, "capacity")?;
        passed = row["valid"] == true;
        report["capacity"] = row;
        report["capacity_input_tokens"] = json!(expected);
    } else {
        let manifest: Value = serde_json::from_str(MANIFEST).unwrap();
        for (index, raw) in FIXTURES.iter().enumerate() {
            let fixture = &manifest["fixtures"][index];
            if fixture["sha256"] != hash(raw.as_bytes()) { return Err("embedded fixture hash mismatch".into()); }
            let id = fixture["task_id"].as_str().unwrap();
            let body = request(serde_json::from_str(raw).unwrap(), false);
            write(&out.join(format!("{id}.request.json")), &body)?;
            let expected = count(&api, &body).map_err(|e| format!("{id}: {e}"))?;
            let warmup = measure(&api, &body, expected, out, &format!("{id}.warmup"))?;
            passed &= warmup["valid"] == true;
            report["requests"].as_array_mut().unwrap().push(json!({
                "id": id, "input_tokens": expected, "warmup": warmup, "samples": [],
            }));
            write(&out.join("run.json"), &report)?;
            for repetition in 0..SAMPLES {
                let row = measure(&api, &body, expected, out, &format!("{id}.sample-{repetition}"))?;
                passed &= row["valid"] == true;
                report["requests"][index]["samples"].as_array_mut().unwrap().push(row);
                write(&out.join("run.json"), &report)?;
            }
            let rate = if report["requests"][index]["warmup"]["valid"] == true {
                median(report["requests"][index]["samples"].as_array().unwrap())
            } else { None };
            report["requests"][index]["median_decode_tokens_per_s"] = json!(rate);
            write(&out.join("run.json"), &report)?;
        }
    }
    report["metrics_after"] = json!(save_metrics("metrics-after.prom"));
    report["card"] = match small_card::run_nvidia_smi().and_then(|s| small_card::parse_nvidia_smi(&s)) {
        Ok((card, _)) => serde_json::to_value(card).unwrap(),
        Err(e) => json!({"error": e}),
    };
    report["finished_at"] = json!(utc_timestamp(unix_now()));
    report["passed"] = json!(passed);
    write(&out.join("run.json"), &report)?;
    if passed { Ok(()) } else { Err(format!("invalid samples retained in {}", out.join("run.json").display())) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reply() -> Value {
        json!({"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [{"function": {"arguments": "{}"}}]}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 8, "prompt_tokens_details": {"cached_tokens": 0},
                "ignis_timing": {"computed_prefill_tokens": 100, "prefill_passes": 1, "output_tokens": 8, "decode_ms": 40.0, "prefill_ms": 10.0, "elapsed_ms": 50.0}}})
    }

    #[test]
    fn a_buffered_call_uses_all_tokens_and_full_server_time() {
        let row = sample(&reply(), 100, 90.0);
        assert_eq!(row["decode_tokens_per_s"], 200.0);
        assert_eq!(row["request_total_ms"], 90.0);
        assert_eq!(median(&[row.clone(), row.clone(), row]), Some(200.0));
    }

    #[test]
    fn failures_cache_hits_empty_outputs_and_truncation_stay_visible() {
        for path in ["/usage/prompt_tokens", "/usage/completion_tokens", "/usage/prompt_tokens_details/cached_tokens", "/usage/ignis_timing/output_tokens", "/usage/ignis_timing/computed_prefill_tokens", "/usage/ignis_timing/decode_ms"] {
            let mut v = reply(); *v.pointer_mut(path).unwrap() = json!(7);
            if path.ends_with("decode_ms") { *v.pointer_mut(path).unwrap() = Value::Null; }
            let row = sample(&v, 100, 90.0);
            assert_eq!(row["valid"], false, "{path}");
            assert_eq!(median(&[row.clone(), row.clone(), row]), None);
        }
        let mut v = reply(); v["choices"][0]["message"] = json!({"content": ""});
        assert_eq!(sample(&v, 100, 90.0)["empty_output"], true);
        v["choices"][0]["message"] = json!({"reasoning_content": "unfinished reasoning"});
        v["choices"][0]["finish_reason"] = json!("length");
        let row = sample(&v, 100, 90.0);
        assert_eq!(row["truncated"], true);
        assert_eq!(row["valid"], true, "truncated timing remains measured and labelled");
    }

    #[test]
    fn frozen_inputs_keep_their_hashes_history_and_tools() {
        let manifest: Value = serde_json::from_str(MANIFEST).unwrap();
        for (i, text) in FIXTURES.iter().enumerate() {
            assert_eq!(manifest["fixtures"][i]["sha256"], hash(text.as_bytes()));
            let body: Value = serde_json::from_str(text).unwrap();
            let sent = request(body.clone(), false);
            assert_eq!(sent["messages"], body["messages"]);
            assert_eq!(sent["tools"], body["tools"]);
            assert_eq!(body["messages"].as_array().unwrap().last().unwrap()["role"], "tool");
            assert_eq!(sent["ignore_eos"], false);
            assert!(sent.get("stop").is_none());
        }
    }
}
