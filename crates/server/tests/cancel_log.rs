//! The `ignis.request.cancelled` line's `cause` (GitHub #327): `timeout` for
//! a request that hit its deadline (`--server-request-timeout`,
//! `--server-stream-timeout`, ADR 0058) on every path that has one (chat
//! completions, with and without a stop sequence, and streamed, the
//! Responses API, streamed, not, and over its WebSocket, and a decision's
//! question), whether the deadline is answered with what was generated or
//! with an error; `consumer` for a client that hung up, `shutdown` for a
//! request still in flight when its model thread stops. Each path with no
//! loop guard, and with one watching the request (a template that starts in
//! the reasoning, as the real ones do), whose own cancel on a stop would
//! otherwise come first.
//!
//! A binary of its own with one test, which installs the process's
//! subscriber: the line is written on the telemetry consumer's task, which a
//! subscriber scoped to the test's thread never sees. The requests never
//! run, so each ends only by its deadline, its client or its model thread:
//! no deadline here races a request finishing.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::{to_bytes, Body};
use axum::http::Request;
use futures_util::{SinkExt, StreamExt};
use ignis_core::decision::{AnswerAlphabet, LabelTokenizer};
use ignis_core::mock::MockCompute;
use ignis_core::{ConcreteScheduler, RequestId, SchedulerConfig, TokenId};
use ignis_logging::{JsonLayer, MemorySink};
use ignis_server::deadline::TimeoutReply;
use ignis_server::engine::Engine;
use ignis_server::loop_guard::{LoopGuardMode, LoopGuardParams};
use ignis_server::decoder::TokenDecoder;
use ignis_server::template::{ChatMessage, RenderedPrompt, SimpleTemplateProvider, TemplateProvider, TemplateRejection};
use ignis_server::thinking::{ThinkingCapabilities, ThinkingOptions};
use ignis_server::Server;
use serde_json::{json, Value};
use tokio_tungstenite::tungstenite::Message;
use tower::ServiceExt;
use tracing_subscriber::layer::SubscriberExt;

#[path = "support/mod.rs"]
mod support;
use support::ReasoningTemplate;

const MODEL: &str = "test-model";

/// The mock's scheduler, except that it never runs what it takes: what it
/// admits stays in flight until it is cancelled.
struct NeverRuns(ConcreteScheduler);

impl ignis_core::Scheduler for NeverRuns {
    fn share_request_ids(&mut self, ids: ignis_core::RequestIds) {
        self.0.share_request_ids(ids)
    }
    fn submit(&mut self, input: ignis_core::RequestInput, class: ignis_core::RequestClass) -> Result<RequestId, ignis_core::SubmitError> {
        self.0.submit(input, class)
    }
    fn cancel(&mut self, request: RequestId) -> bool {
        self.0.cancel(request)
    }
    fn advance(&mut self) -> Vec<ignis_core::SchedEvent> {
        Vec::new()
    }
    fn is_idle(&self) -> bool {
        true
    }
    fn model_id(&self) -> &str {
        self.0.model_id()
    }
    fn max_sequence_tokens(&self) -> u32 {
        self.0.max_sequence_tokens()
    }
    fn mode(&self) -> ignis_core::EngineMode {
        self.0.mode()
    }
    fn occupancy(&self) -> ignis_core::Occupancy {
        self.0.occupancy()
    }
}

/// An engine over a scheduler that never runs what it takes.
fn never_runs() -> Engine {
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig { model: MODEL.into(), max_in_flight: 8, ..SchedulerConfig::default() },
        Arc::new(MockCompute::new()),
    );
    Engine::new(Box::new(NeverRuns(scheduler)))
}

/// Just enough tokenizer to name two options: `/v1/decide` refuses a load
/// whose tokenizer can label none.
struct TwoLetters;

impl LabelTokenizer for TwoLetters {
    fn encode(&self, text: &str) -> Option<Vec<TokenId>> {
        match text {
            "A" => Some(vec![1]),
            "B" => Some(vec![2]),
            _ => None,
        }
    }

    fn decode(&self, ids: &[TokenId]) -> Option<String> {
        match ids {
            [1] => Some("A".to_owned()),
            [2] => Some("B".to_owned()),
            _ => None,
        }
    }
}

/// [`SimpleTemplateProvider`] with a two-label answer alphabet.
struct DecidingTemplate;

impl TemplateProvider for DecidingTemplate {
    fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        options: &ThinkingOptions,
        tools: &[Value],
    ) -> Result<RenderedPrompt, TemplateRejection> {
        SimpleTemplateProvider.apply_chat_template(messages, options, tools)
    }

    fn answer_alphabet(&self) -> AnswerAlphabet {
        AnswerAlphabet::from_tokenizer(&TwoLetters)
    }

    fn render_tokens(&self, tokens: &[TokenId]) -> String {
        SimpleTemplateProvider.render_tokens(tokens)
    }

    fn thinking_capabilities(&self) -> ThinkingCapabilities {
        SimpleTemplateProvider.thinking_capabilities()
    }

    fn token_decoder(&self) -> Box<dyn TokenDecoder> {
        SimpleTemplateProvider.token_decoder()
    }
}

/// Every `ignis.request.cancelled` line `log` holds, as `(request_id, cause)`.
fn cancelled(log: &MemorySink) -> Vec<(u64, String)> {
    log.lines()
        .iter()
        .map(|line| serde_json::from_str::<Value>(line).expect("json"))
        .filter(|event| event["event_name"] == "ignis.request.cancelled")
        .map(|event| {
            let attributes = &event["attributes"];
            (attributes["request_id"].as_u64().unwrap(), attributes["cause"].as_str().unwrap().to_owned())
        })
        .collect()
}

/// The cause of the `n`th cancelled line, once `log` holds it.
async fn nth_cause(log: &MemorySink, n: usize) -> String {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let lines = cancelled(log);
        if let Some((_, cause)) = lines.get(n) {
            return cause.clone();
        }
        assert!(Instant::now() < deadline, "only {lines:?} after waiting for line {n}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

fn post(path: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

/// Every path once on a server over `template` with the loop guard in `mode`
/// and a passed deadline answered as `reply` says, the shutdown last; `seen`
/// is how many cancelled lines `log` held before, and the count after is
/// returned. Each request logs one line.
async fn every_cause(template: Box<dyn TemplateProvider>, mode: LoopGuardMode, reply: TimeoutReply, log: &MemorySink, seen: usize) -> usize {
    let engine = never_runs();
    let server = Server::new(engine.clone(), template)
        .with_request_timeout(Duration::from_millis(100))
        .with_stream_timeout(Duration::from_millis(100))
        .with_timeout_reply(reply)
        .with_loop_guard(LoopGuardParams { mode, ..LoopGuardParams::default() })
        .with_seedless_seed(0);
    let app = server.app();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(server.clone().serve_on_until(listener, None, std::future::pending()));
    let chat = json!({ "model": MODEL, "messages": [{ "role": "user", "content": "hi" }], "max_tokens": 8 });
    let mut n = seen;

    // Past the deadline: chat completions, with a stop sequence (its own
    // collection path), and the Responses API.
    let mut stopped = chat.clone();
    stopped["stop"] = json!(["zzz"]);
    for (path, body) in [
        ("/v1/chat/completions", chat.clone()),
        ("/v1/chat/completions", stopped),
        ("/v1/responses", json!({ "model": MODEL, "input": "hi", "max_output_tokens": 8 })),
    ] {
        let response = app.clone().oneshot(post(path, body)).await.unwrap();
        let (status, stopped_by) = (response.status(), response.headers().get("ignis-stopped-by").cloned());
        match reply {
            TimeoutReply::Partial => assert!(status == 200 && stopped_by.is_some_and(|v| v == "request_timeout"), "{path}: {status}"),
            TimeoutReply::Error => assert_eq!(status, 504, "{path}"),
        }
        assert_eq!(nth_cause(log, n).await, "timeout", "{path}");
        n += 1;
    }
    // Streamed past the deadline: a chat stream ends `length` or with its
    // error chunk, a response `incomplete` or `failed`.
    let mut streamed_chat = chat.clone();
    streamed_chat["stream"] = json!(true);
    let streamed_response = json!({ "model": MODEL, "input": "hi", "max_output_tokens": 8, "stream": true });
    let ended = match reply {
        TimeoutReply::Partial => ["\"finish_reason\":\"length\"", "event: response.incomplete"],
        TimeoutReply::Error => ["\"request_timeout\"", "\"request_timeout\""],
    };
    for ((path, body), ended) in [("/v1/chat/completions", streamed_chat), ("/v1/responses", streamed_response)].into_iter().zip(ended) {
        let response = app.clone().oneshot(post(path, body)).await.unwrap();
        assert_eq!(response.status(), 200);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert!(String::from_utf8_lossy(&body).contains(ended), "{path}: {body:?}");
        assert_eq!(nth_cause(log, n).await, "timeout", "streamed {path}");
        n += 1;
    }
    // And one over the WebSocket.
    let (mut socket, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/v1/responses")).await.unwrap();
    let create = json!({ "type": "response.create", "model": MODEL, "input": "hi", "max_output_tokens": 8 });
    socket.send(Message::Text(create.to_string().into())).await.unwrap();
    loop {
        let frame = tokio::time::timeout(Duration::from_secs(10), socket.next()).await.expect("a frame").unwrap().unwrap();
        let Message::Text(text) = frame else { continue };
        let event: Value = serde_json::from_str(&text).unwrap();
        match (reply, event["type"].as_str()) {
            (TimeoutReply::Error, Some("response.failed")) => {
                assert_eq!(event["response"]["error"]["code"], "request_timeout", "{event}");
                break;
            }
            (TimeoutReply::Partial, Some("response.incomplete")) => {
                assert_eq!(event["response"]["incomplete_details"]["reason"], "max_output_tokens", "{event}");
                break;
            }
            (_, Some("response.failed" | "response.incomplete" | "response.completed")) => panic!("{reply:?}: {event}"),
            _ => {}
        }
    }
    assert_eq!(nth_cause(log, n).await, "timeout", "WebSocket /v1/responses");
    n += 1;

    // A client that hangs up on a stream.
    let mut streamed = chat.clone();
    streamed["stream"] = json!(true);
    let response = app.clone().oneshot(post("/v1/chat/completions", streamed.clone())).await.unwrap();
    assert_eq!(response.status(), 200);
    drop(response);
    assert_eq!(nth_cause(log, n).await, "consumer", "a stream dropped");
    n += 1;

    // A stream still in flight when the model thread stops (a switch past
    // its drain window).
    let response = app.clone().oneshot(post("/v1/chat/completions", streamed)).await.unwrap();
    assert_eq!(response.status(), 200);
    engine.shutdown();
    assert_eq!(nth_cause(log, n).await, "shutdown", "a stream in flight at shutdown");
    let _ = to_bytes(response.into_body(), usize::MAX).await;
    n += 1;

    // One line per request: a second cancel (a guard and a response both
    // letting go) logs nothing. This engine's ids start again at 0.
    let lines = cancelled(log);
    let mut ids: Vec<u64> = lines[seen..].iter().map(|(id, _)| *id).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!((lines.len(), ids.len()), (n, n - seen), "{lines:?}");
    n
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_cancelled_request_says_why_it_was_cancelled() {
    let log = Arc::new(MemorySink::new());
    tracing::subscriber::set_global_default(tracing_subscriber::registry().with(JsonLayer::new(log.clone())))
        .expect("the only subscriber this binary installs");

    let mut seen = 0;
    for reply in [TimeoutReply::Error, TimeoutReply::Partial] {
        seen = every_cause(Box::new(SimpleTemplateProvider), LoopGuardMode::Off, reply, &log, seen).await;
        seen = every_cause(Box::new(ReasoningTemplate), LoopGuardMode::Rewind, reply, &log, seen).await;
    }

}
