//! GitHub #251: the `/v1` surface documents itself. The document is
//! generated from the handler annotations by the same call that registers
//! the routes, so these tests are what stops the two drifting — a route
//! added with `.route()` alone fails [`the_document_lists_exactly_the_v1_surface`],
//! and a monitoring path that leaks in fails
//! [`the_document_carries_no_monitoring_path`].
//!
//! The page, the document and their assets answer without a key even on a
//! keyed server (the API's shape is not its load); the handler routes still
//! do not.

use std::collections::BTreeSet;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode, header};
use http_body_util::BodyExt;
use ignis_core::{ConcreteScheduler, MockCompute, SchedulerConfig};
use ignis_server::Server;
use ignis_server::config::ApiKey;
use ignis_server::engine::Engine;
use ignis_server::template::SimpleTemplateProvider;
use tower::ServiceExt;

const MODEL: &str = "test-model";
const KEY: &str = "sk-test-key";

fn server() -> Server {
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig { model: MODEL.into(), ..SchedulerConfig::default() },
        Arc::new(MockCompute::new()),
    );
    Server::new(Engine::new(Box::new(scheduler)), Box::new(SimpleTemplateProvider))
}

async fn get(app: &axum::Router, uri: &str) -> axum::response::Response {
    let request = Request::builder().uri(uri).body(Body::empty()).unwrap();
    app.clone().oneshot(request).await.unwrap()
}

async fn body_bytes(response: axum::response::Response) -> Vec<u8> {
    response.into_body().collect().await.unwrap().to_bytes().to_vec()
}

fn header_value(response: &axum::response::Response, name: header::HeaderName) -> String {
    response
        .headers()
        .get(name)
        .map(|v| v.to_str().unwrap().to_owned())
        .unwrap_or_default()
}

/// The document itself, as the JSON a client generator reads. Asserting
/// over the serialized form rather than over `utoipa`'s types keeps these
/// tests about the contract, not about the generator's internals.
fn document() -> serde_json::Value {
    serde_json::to_value(ignis_server::api::openapi()).unwrap()
}

/// The document's paths and the methods each one serves.
fn documented() -> BTreeSet<String> {
    document()["paths"]
        .as_object()
        .expect("paths")
        .iter()
        .flat_map(|(path, item)| {
            item.as_object()
                .expect("a path item")
                .keys()
                .map(move |method| format!("{method} {path}"))
                .collect::<Vec<_>>()
        })
        .collect()
}

// ── the document ─────────────────────────────────────────────────────────

#[test]
fn the_document_lists_exactly_the_v1_surface() {
    // Set equality, both ways: a new `/v1` route that ships without its
    // `#[utoipa::path]` fails here, and so does a path entry whose route
    // was removed.
    let expected: BTreeSet<String> = [
        "get /v1/models",
        // Spec model-switch/01: replace the loaded model.
        "post /v1/models/switch",
        // Spec playground/05: the models a switch can go to or fetch, and
        // what each configuration field is.
        "get /v1/models/catalog",
        // GitHub #380: remove a catalog entry's files from the models folder.
        "post /v1/models/remove",
        // GitHub #376: download a catalog entry in the background.
        "post /v1/models/download",
        // GitHub #377: stop it, keeping its partial files.
        "post /v1/models/download/cancel",
        "get /v1/config/fields",
        // Spec playground/05 follow-up 2: where each of those values comes
        // from.
        "get /v1/config/origins",
        // Spec config-v2/02: the running configuration, read and changed.
        "get /v1/config",
        "patch /v1/config",
        "post /v1/chat/completions",
        // GitHub #285: the prompt counted without being served.
        "post /v1/tokenize",
        "post /v1/detokenize",
        "post /v1/responses",
        // GitHub #282: the Responses API's WebSocket mode, deliberately.
        "get /v1/responses",
        // Spec image-model/05: the image model's routes.
        // Spec speech/02: the speech model's transcription.
        // Spec speech/03: its Realtime transcription session, a WebSocket.
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    assert_eq!(documented(), expected);
}

/// Spec playground/05 AC 9: the Server view's routes are in the document
/// with their responses, and the rows they return are described.
#[test]
fn the_server_view_routes_are_documented_with_their_responses() {
    let document = document();
    for (path, list, ok) in [("/v1/models/catalog", "CatalogList", "CatalogRow"), ("/v1/config/fields", "FieldList", "FieldDeclaration"), ("/v1/config/origins", "OriginList", "OriginRow")] {
        let get = &document["paths"][path]["get"];
        for status in ["200", "401"] {
            assert!(get["responses"][status].is_object(), "{path} documents {status}");
        }
        assert_eq!(
            get["responses"]["200"]["content"]["application/json"]["schema"]["$ref"],
            format!("#/components/schemas/{list}"),
            "{path}"
        );
        assert!(document["components"]["schemas"][ok]["properties"].is_object(), "{ok}");
    }
    // An origin's source is one of six, each named; no configuration is a 501.
    assert_eq!(
        document["components"]["schemas"]["OriginSource"]["enum"],
        serde_json::json!(["default", "profile", "file", "env", "flag", "live"])
    );
    assert!(document["paths"]["/v1/config/origins"]["get"]["responses"]["501"].is_object());
    // The switch takes a model alone, and says what a 404 is.
    let switch = &document["paths"]["/v1/models/switch"]["post"];
    assert!(switch["responses"]["404"].is_object());
    let request = &document["components"]["schemas"]["SwitchRequest"];
    assert_eq!(request["required"], serde_json::json!(["model"]), "artifact is optional");
    // `GET /v1/models` carries how the last switch ended.
    assert!(document["components"]["schemas"]["ModelList"]["properties"]["last_switch"].is_object());
    assert!(document["components"]["schemas"]["LastSwitchInfo"]["properties"]["outcome"].is_object());
}

/// Spec playground/05 AC 22: `GET /v1/config` documents its `format` query,
/// the two media types it answers in, and the `400` of any other format.
#[test]
fn the_config_route_documents_its_format_and_both_media_types() {
    let document = document();
    let get = &document["paths"]["/v1/config"]["get"];
    let parameters = get["parameters"].as_array().expect("parameters");
    let format = parameters.iter().find(|parameter| parameter["name"] == "format").expect("the format query");
    assert_eq!((format["in"].clone(), format["required"].clone()), (serde_json::json!("query"), serde_json::json!(false)));
    // The two formats are the enum the handler matches on, not a sentence.
    let schema = &format["schema"];
    let listed = schema.get("enum").or_else(|| schema["oneOf"].as_array()?.iter().find_map(|any| any.get("enum")));
    assert_eq!(listed, Some(&serde_json::json!(["json", "yaml"])), "{format}");
    let content = &get["responses"]["200"]["content"];
    assert_eq!(content["application/json"]["schema"]["$ref"], "#/components/schemas/ConfigDocument");
    assert_eq!(content["application/yaml"]["schema"]["type"], "string");
    assert!(get["responses"]["400"].is_object());
}

/// ADR 0058: the deadline is documented where a client reads it — the
/// header a cut whole answer carries, declared on the `200` of chat
/// completions and `/v1/responses`, and the two deadlines, `0` and the two
/// replies in their descriptions, the WebSocket mode's included.
#[test]
fn the_deadline_is_documented_on_every_route_that_has_one() {
    let document = document();
    for path in ["/v1/chat/completions", "/v1/responses"] {
        let post = &document["paths"][path]["post"];
        let ok = &post["responses"]["200"];
        assert_eq!(ok["headers"]["ignis-stopped-by"]["schema"]["type"], "string", "{path}: {ok}");
        let description = post["description"].as_str().unwrap();
        for words in ["--server-request-timeout", "--server-stream-timeout", "--server-timeout-reply", "`0`", "request_timeout"] {
            assert!(description.contains(words), "{path} names {words}: {description}");
        }
    }
    // A chat request the engine let go of is its engine error, never the 504.
    let engine_error = document["paths"]["/v1/chat/completions"]["post"]["responses"]["500"]["description"].as_str().unwrap_or_default();
    assert!(engine_error.contains("engine_error"), "{engine_error}");
    let responses = document["paths"]["/v1/responses"]["post"]["description"].as_str().unwrap();
    assert!(responses.contains("`max_output_tokens` or the deadline"), "{responses}");
    let socket = document["paths"]["/v1/responses"]["get"]["description"].as_str().unwrap();
    for words in ["--server-stream-timeout", "`0`", "response.incomplete", "request_timeout", "--server-timeout-reply"] {
        assert!(socket.contains(words), "the socket names {words}: {socket}");
    }
}

#[test]
fn the_counting_routes_carry_their_schemas() {
    // GitHub #285: the request and response shapes are in the document, not
    // only the two paths.
    let document = document();
    for schema in ["TokenizeRequest", "TokenizeResponse", "DetokenizeRequest", "DetokenizeResponse"] {
        assert!(document["components"]["schemas"][schema].is_object(), "{schema}");
    }
    assert_eq!(
        document["paths"]["/v1/tokenize"]["post"]["requestBody"]["content"]["application/json"]["schema"]["$ref"],
        "#/components/schemas/TokenizeRequest"
    );
}

#[test]
fn the_models_list_documents_the_servable_models_and_the_active_entry() {
    // Spec model-switch/03 AC 12: what is listed, and that `data[0]` is the
    // active model.
    let document = document();
    let list = &document["paths"]["/v1/models"]["get"];
    assert!(list["summary"].as_str().unwrap().contains("servable"), "{list}");
    let description = list["description"].as_str().unwrap();
    for says in ["`data[0]` is always the model this server loaded", "`download.path`", "`switch.known_models`", "`switch.allow_implicit` false"] {
        assert!(description.contains(says), "{says}: {description}");
    }
    let entry = &document["components"]["schemas"]["ModelInfo"];
    assert!(entry["properties"]["active"]["description"].as_str().unwrap().contains("data[0]"), "{entry}");
    assert!(entry["properties"]["kind"]["description"].as_str().unwrap().contains("family's"), "{entry}");
    let required: Vec<&str> = entry["required"].as_array().unwrap().iter().map(|key| key.as_str().unwrap()).collect();
    assert!(required.contains(&"active") && !required.contains(&"kind"), "{entry}");
}

/// ADR 0064 (GitHub #336): every route that names a model documents the
/// card-class refusal — `409`, or `422` on the decision endpoint as its other
/// model refusals — and both model lists say what `available` and `reason`
/// are.
#[test]
fn the_card_class_refusal_and_availability_are_documented() {
    let document = document();
    for (path, status) in [
        ("/v1/chat/completions", "409"),
        ("/v1/responses", "409"),
        ("/v1/tokenize", "409"),
        ("/v1/models/switch", "409"),
        ("/v1/models/download", "409"),
    ] {
        let said = document["paths"][path]["post"]["responses"][status]["description"].as_str().unwrap_or_default();
        assert!(said.contains("model_above_card_class"), "{path} {status}: {said}");
    }
    for (schema, path) in [("ModelInfo", "/v1/models"), ("CatalogRow", "/v1/models/catalog")] {
        let entry = &document["components"]["schemas"][schema];
        let required: Vec<&str> = entry["required"].as_array().unwrap().iter().map(|key| key.as_str().unwrap()).collect();
        assert!(required.contains(&"available") && !required.contains(&"reason"), "{schema}: {entry}");
        assert!(entry["properties"]["available"]["description"].as_str().unwrap().contains("card_class"), "{schema}: {entry}");
        assert!(entry["properties"]["reason"]["description"].as_str().unwrap().contains("available: false"), "{schema}: {entry}");
        let description = document["paths"][path]["get"]["description"].as_str().unwrap();
        assert!(description.contains("`available: false`") && description.contains("ADR 0064"), "{path}: {description}");
    }
}

#[test]
fn every_refused_field_is_named_in_its_operation_description() {
    // Spec server/09: driven off the table the validator reads, so a refusal
    // cannot be added without the document saying so.
    use ignis_server::openai_fields::{table, Surface};
    let document = document();
    for (path, surface) in [
        ("/v1/chat/completions", Surface::Chat),
        ("/v1/responses", Surface::Responses),
    ] {
        let description = document["paths"][path]["post"]["description"].as_str().unwrap_or_default();
        for rule in table(surface) {
            assert!(description.contains(&format!("`{}`", rule.name)), "{path}: `{}` not named", rule.name);
            if let Some(reason) = rule.refusal {
                assert!(description.contains(reason), "{path}: `{}`'s refusal not described", rule.name);
            }
        }
    }
    // What the chat operation honours beyond the classic fields.
    let chat = document["paths"]["/v1/chat/completions"]["post"]["description"].as_str().unwrap();
    for honoured in ["`max_completion_tokens`", "`stop`", "`usage.prompt_tokens_details.cached_tokens`"] {
        assert!(chat.contains(honoured), "{honoured}");
    }
}

#[test]
fn the_document_names_what_the_server_does_not_serve() {
    let description = document()["info"]["description"].as_str().unwrap_or_default().to_owned();
    for choice in ["embeddings", "`/v1/completions`", "`/v1/batches`", "audio out", "`/v1/audio/translations`", "LoRA", "a second model on the card"] {
        assert!(description.contains(choice), "`{choice}` missing: {description}");
    }
}

#[test]
fn the_document_carries_no_monitoring_path() {
    // "Not for the monitoring" (GitHub #251), made checkable: the
    // Prometheus exposition is ADR 0017's contract, served on its own
    // listener, and the Playground's pages are assets rather than API.
    for path in document()["paths"].as_object().expect("paths").keys() {
        assert!(
            !path.contains("metrics") && !path.starts_with("/ui"),
            "the document must not carry a monitoring or Playground path: {path}",
        );
    }
}

#[test]
fn the_error_envelope_is_one_component() {
    // Shared, not inlined per operation: a client handles failures on this
    // surface with one type.
    let document = document();
    assert!(document["components"]["schemas"]["ApiError"].is_object());
    let four_hundred = document["paths"]["/v1/chat/completions"]["post"]["responses"]["400"]
        ["content"]["application/json"]["schema"]["$ref"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    assert_eq!(four_hundred, "#/components/schemas/ApiError");
}

#[test]
fn the_streaming_half_is_documented_as_its_own_content_type() {
    // `stream: true` answers SSE chunks, not the single JSON body: a
    // client generator that sees only one of the two writes the wrong
    // reader.
    let document = document();
    let ok = &document["paths"]["/v1/chat/completions"]["post"]["responses"]["200"]["content"];
    assert!(ok["application/json"].is_object(), "{ok}");
    assert!(ok["text/event-stream"].is_object(), "{ok}");
}

#[test]
fn a_forced_thinking_close_is_documented_as_an_optional_field_of_each_response_shape() {
    // Spec server/08: the field is absent when the close was not forced, so
    // a generated client must not require it.
    let document = document();
    for schema in ["CompletionChoice", "ChunkChoice", "ResponseObject"] {
        let schema = &document["components"]["schemas"][schema];
        assert!(schema["properties"]["thinking_budget_forced_at"].is_object(), "{schema}");
        let required = schema["required"].as_array().cloned().unwrap_or_default();
        assert!(!required.contains(&serde_json::json!("thinking_budget_forced_at")), "{schema}");
    }
}

// ── the routes ───────────────────────────────────────────────────────────

#[tokio::test]
async fn v1_redirects_to_the_page() {
    let app = server().app();
    for uri in ["/v1", "/v1/", "/v1/docs"] {
        let response = get(&app, uri).await;
        assert_eq!(response.status(), StatusCode::TEMPORARY_REDIRECT, "{uri}");
        assert_eq!(header_value(&response, header::LOCATION), "/v1/docs/", "{uri}");
    }
}

#[tokio::test]
async fn the_page_and_its_assets_come_from_this_binary() {
    // The vendored distribution, not a CDN: a server with no route to the
    // internet serves the same page.
    let app = server().app();
    let page = get(&app, "/v1/docs/").await;
    assert_eq!(page.status(), StatusCode::OK);
    assert_eq!(
        header_value(&page, header::CONTENT_TYPE),
        "text/html; charset=utf-8",
    );
    let html = String::from_utf8(body_bytes(page).await).unwrap();
    assert!(html.contains("/v1/docs/swagger-ui-bundle.js"), "{html}");
    assert!(html.contains("/v1/openapi.json"), "{html}");
    assert!(!html.contains("//cdn."), "the page must name no CDN: {html}");

    for (file, content_type) in [
        ("swagger-ui-bundle.js", "text/javascript; charset=utf-8"),
        ("swagger-ui.css", "text/css; charset=utf-8"),
    ] {
        let response = get(&app, &format!("/v1/docs/{file}")).await;
        assert_eq!(response.status(), StatusCode::OK, "{file}");
        assert_eq!(header_value(&response, header::CONTENT_TYPE), content_type, "{file}");
        assert!(!body_bytes(response).await.is_empty(), "{file}");
    }

    // The bundle's own third-party notices, served beside it as its
    // licence asks.
    assert_eq!(
        get(&app, "/v1/docs/swagger-ui-bundle.js.LICENSE.txt").await.status(),
        StatusCode::OK,
    );
    // Exact table lookup: nothing else is reachable under the page.
    assert_eq!(get(&app, "/v1/docs/swagger-ui.js").await.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn the_document_is_served_as_openapi_3_1() {
    let app = server().app();
    let response = get(&app, "/v1/openapi.json").await;
    assert_eq!(response.status(), StatusCode::OK);
    assert!(header_value(&response, header::CONTENT_TYPE).starts_with("application/json"));
    let document: serde_json::Value = serde_json::from_slice(&body_bytes(response).await).unwrap();
    assert!(
        document["openapi"].as_str().unwrap_or_default().starts_with("3.1"),
        "{}",
        document["openapi"],
    );
    assert_eq!(document["info"]["version"], env!("CARGO_PKG_VERSION"));
    assert!(document["paths"]["/v1/chat/completions"]["post"].is_object());
    assert!(
        document["components"]["securitySchemes"]["bearerAuth"]["scheme"] == "bearer",
        "{}",
        document["components"]["securitySchemes"],
    );
}

#[tokio::test]
async fn the_reference_is_open_on_a_keyed_server() {
    // A browser pointed at `/v1` has no way to set an `Authorization`
    // header, so a keyed server that answered 401 here would publish a
    // reference nobody could read. The handler routes stay gated.
    let app = server().with_api_key(ApiKey::new(KEY.to_owned())).app();
    for uri in ["/v1", "/v1/docs/", "/v1/docs/swagger-ui.css", "/v1/openapi.json"] {
        let status = get(&app, uri).await.status();
        assert!(
            status == StatusCode::OK || status == StatusCode::TEMPORARY_REDIRECT,
            "{uri} answered {status}",
        );
    }
    assert_eq!(get(&app, "/v1/models").await.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn the_preflight_still_answers_on_every_v1_route() {
    // The `OPTIONS` handlers are attached beside the documented ones
    // rather than through `routes!`, so this is what says they survived
    // the move to `OpenApiRouter`.
    let app = server().app();
    for uri in [
        "/v1/models",
        "/v1/chat/completions",
        "/v1/responses",
    ] {
        let request = Request::builder()
            .method(Method::OPTIONS)
            .uri(uri)
            .body(Body::empty())
            .unwrap();
        let response = app.clone().oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK, "{uri}");
    }
}

#[test]
fn the_field_declaration_schema_names_exactly_the_keys_a_declaration_has() {
    // ADR 0046: the route builds each object with `FieldMeta::declaration`;
    // the schema restating it by hand must not drift from it silently.
    let document = document();
    let schema: BTreeSet<String> =
        document["components"]["schemas"]["FieldDeclaration"]["properties"].as_object().expect("properties").keys().cloned().collect();
    let declared: BTreeSet<String> = ignis_server::config::schema::all_fields().next().expect("a field").declaration().keys().cloned().collect();
    assert_eq!(schema, declared);
}
