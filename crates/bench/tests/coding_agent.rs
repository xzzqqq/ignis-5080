//! Exercise the reusable runner's request isolation, repetitions and evidence.
use std::sync::{Arc, Mutex};

use axum::{extract::State, routing::{get, post}, Json, Router};
use ignis_bench::coding_agent;
use serde_json::{json, Value};

#[test]
fn a_failed_sample_is_archived_and_never_hidden_by_the_median() {
    let seen = Arc::new(Mutex::new(Vec::<Value>::new()));
    let capture = seen.clone();
    let (url_tx, url_rx) = std::sync::mpsc::channel();
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel();
    let worker = std::thread::spawn(move || {
        tokio::runtime::Runtime::new().unwrap().block_on(async move {
            async fn complete(State(seen): State<Arc<Mutex<Vec<Value>>>>, Json(body): Json<Value>) -> Json<Value> {
                let mut seen = seen.lock().unwrap();
                seen.push(body);
                // The middle sample of the first request has an unproven cache.
                let cached = if seen.len() == 3 { 64 } else { 0 };
                Json(json!({"choices": [{"finish_reason": "tool_calls", "message": {
                    "reasoning_content": "analysis", "tool_calls": [{"function": {"name": "bash", "arguments": "{\"command\":\"DO NOT EXECUTE\"}"}}]
                }}], "usage": {"prompt_tokens": 100, "completion_tokens": 10,
                    "prompt_tokens_details": {"cached_tokens": cached},
                    "ignis_timing": {"computed_prefill_tokens": 100, "prefill_passes": 1,
                        "output_tokens": 10, "decode_ms": 50.0, "prefill_ms": 20.0, "elapsed_ms": 70.0}}}))
            }
            let app = Router::new()
                .route("/v1/chat/completions", post(complete).with_state(capture))
                .route("/v1/tokenize", post(|| async { Json(json!({"count": 100})) }))
                .route("/v1/models", get(|| async { Json(json!({"data": [{"id": "test", "max_model_len": 262144}]})) }))
                .route("/v1/models/catalog", get(|| async { Json(json!({"data": [{"id": "test", "family": "qwen38", "loaded": true}]})) }))
                .route("/v1/config/origins", get(|| async { Json(json!({"data": []})) }))
                .route("/v1/config", get(|| async { Json(json!({
                    "model": {"loop_guard": "off"}, "reuse": {"prompt": false}, "server": {"metrics": false},
                })) }));
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            url_tx.send(format!("http://{}", listener.local_addr().unwrap())).unwrap();
            axum::serve(listener, app).with_graceful_shutdown(async { let _ = stop_rx.await; }).await.unwrap();
        });
    });
    let out = std::env::temp_dir().join(format!("ignis-coding-agent-{}", std::process::id()));
    assert!(!out.exists());
    let result = coding_agent::run(&url_rx.recv().unwrap(), &out, false);
    stop_tx.send(()).unwrap();
    worker.join().unwrap();
    assert!(result.is_err());
    let report: Value = serde_json::from_slice(&std::fs::read(out.join("run.json")).unwrap()).unwrap();
    let sent = seen.lock().unwrap();
    assert_eq!(sent.len(), 12, "warmup plus three repeats for each frozen request");
    for i in 0..3 {
        let fixture: Value = serde_json::from_str(coding_agent::FIXTURES[i]).unwrap();
        for body in &sent[i * 4..i * 4 + 4] {
            assert_eq!(body["messages"], fixture["messages"]);
            assert_eq!(body["tools"], fixture["tools"]);
        }
        let rows = report["requests"][i]["samples"].as_array().unwrap();
        assert_eq!(rows.len(), 3);
        for row in rows { assert!(out.join(row["response_file"].as_str().unwrap()).is_file()); }
    }
    assert!(report["requests"][0]["median_decode_tokens_per_s"].is_null());
    assert_eq!(report["requests"][1]["median_decode_tokens_per_s"], 200.0);
    assert_eq!(report["requests"][0]["samples"][1]["valid"], false);
    assert_eq!(report["passed"], false);
    assert!(coding_agent::run("http://127.0.0.1:1", &out, false).is_err(), "cannot overwrite a run");
    std::fs::remove_dir_all(out).unwrap();
}
