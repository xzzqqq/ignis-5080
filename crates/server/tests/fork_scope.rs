//! The product boundary of the 5080 fork, exercised through its public seams.
use std::sync::Arc;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use ignis_artifact::{ArtifactIdentity, ContainerWriter, ResourceEncoding};
use ignis_core::{ConcreteScheduler, MockCompute, SchedulerConfig};
use ignis_server::{config, Server};
use ignis_server::engine::Engine;
use ignis_server::template::SimpleTemplateProvider;
use tower::ServiceExt;

#[tokio::test]
async fn retired_routes_are_absent_while_chat_and_vision_contracts_remain() {
    let scheduler = ConcreteScheduler::with_config(SchedulerConfig::default(), Arc::new(MockCompute::new()));
    let app = Server::new(Engine::new(Box::new(scheduler)), Box::new(SimpleTemplateProvider)).app();
    let document = serde_json::to_value(ignis_server::api::openapi()).unwrap();
    for path in ["/v1/decide", "/v1/systemone", "/v1/images/generations", "/v1/images/edits", "/v1/audio/transcriptions", "/v1/realtime"] {
        let request = Request::builder().method("POST").uri(path).header("content-type", "application/json").body(Body::from("{}")).unwrap();
        assert_eq!(app.clone().oneshot(request).await.unwrap().status(), StatusCode::NOT_FOUND, "{path}");
        assert!(document["paths"].get(path).is_none(), "{path}");
    }
    for path in ["/v1/chat/completions", "/v1/responses", "/v1/tokenize", "/v1/detokenize"] {
        assert!(document["paths"].get(path).is_some(), "{path}");
    }
    let config::ConfigOutcome::Config(options) = config::resolve(&[], |_| None).unwrap() else { panic!("config") };
    let options = options.for_family(ignis_core::compute::ModelFamily::Qwen38_27b).unwrap();
    assert_eq!(options.model, "qwen3.8-27b-16g");
    assert_eq!(options.decode_lanes, Some(2));
    assert_eq!(options.vision.unwrap().max_tokens(), 4096);
    assert!(options.speculation.is_some());
}

#[test]
fn removed_config_flags_cannot_silently_change_the_service() {
    for flag in ["--speech-enabled", "--diarizer-enabled", "--image-max-jobs", "--server-expose", "--ngram-hot-bytes", "--spec-draft-rows", "--vram-pinned-expert-bytes", "--qwen38flashnext-model-max-context"] {
        assert!(config::resolve(&[flag.to_owned(), "1".to_owned()], |_| None).is_err(), "{flag}");
    }
    assert!(config::resolve(&["--spec-backend".into(), "mtp".into()], |_| None).is_err());
}

#[test]
fn a_retired_artifact_is_refused_before_sidecar_or_gpu_work() {
    let config::ConfigOutcome::Config(options) = config::resolve(&[], |_| None).unwrap() else { panic!("config") };
    let dir = std::env::temp_dir().join(format!("ignis-fork-scope-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for id in ["qwen3.8-flash-next", "ignis-image-2.1-turbo", "unknown"] {
        let path = dir.join(format!("{id}.ninfer"));
        let identity = ArtifactIdentity { model_id: id.into(), weights_id: "test".into() };
        let mut writer = ContainerWriter::create(&path, identity, 4096).unwrap();
        writer.append_resource("test/r", ResourceEncoding::RawBytesV1, 1, &mut &b"x"[..]).unwrap();
        writer.finish().unwrap();
        assert!(matches!(ignis_server::load::prepare_model(&options, &path), Err(ignis_server::load::LoadError::ModelMismatch(_))), "{id}");
    }
    std::fs::remove_dir_all(dir).unwrap();
}
