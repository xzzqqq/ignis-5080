//! A **Removal** through the API (GitHub #380, spec #374, ADR 0071), over the
//! mock-backed router with a temporary `download.path` holding small fake
//! files (`support/repo.rs`'s catalog): no network, no GPU, no real model.
//! `POST /v1/models/remove` deletes a catalog entry's files and partial
//! files; what is left under the models folder, and the entry's row in
//! `GET /v1/models/catalog`, are the outcome. A download the removal must
//! refuse is held mid-transfer by holding the local endpoint's response,
//! never by a timer.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use ignis_server::config::Config;
use ignis_server::download_http::Downloads;
use ignis_server::model_switch::{CatalogModels, Switcher};
use ignis_server::Server;
use serde_json::{json, Value};
use tower::ServiceExt;

#[path = "support/switch.rs"]
mod switch_support;
use switch_support::MockLoader;

#[path = "support/repo.rs"]
mod repo;
use repo::{catalog_served, files_of, Repo};

#[path = "support/model_routes.rs"]
mod routes;
use routes::{catalog_row, config, files_in, model_download, send, until_row, until_status, PATIENT};

async fn remove(app: &axum::Router, model: &str) -> (StatusCode, Value) {
    send(app, Method::POST, "/v1/models/remove", Some(json!({ "model": model }))).await
}

/// A server on `loaded`, switching to the catalog `models`, downloading as
/// `config` says.
fn server_on(loader: &Arc<MockLoader>, loaded: &str, models: CatalogModels, config: &Config) -> Server {
    Server::from_active(loader.model(loaded))
        .with_switcher(Switcher::new(Arc::clone(loader) as _, PATIENT).with_catalog(models, true))
        .with_downloads(Downloads::of(config, None))
}

/// Put every file of `id` under `dir`, as a finished download leaves them.
fn on_disk(dir: &Path, id: &str) {
    let files = files_of(id);
    std::fs::write(dir.join(&files.sidecar.0), &files.sidecar.1).unwrap();
    std::fs::write(dir.join(&files.artifact.0), &files.artifact.1).unwrap();
}


/// AC 1: a removal deletes every file of the entry and answers once they
/// are gone, with the bytes it freed; the row reads absent, and the model is
/// no longer one `GET /v1/models` lists.
#[tokio::test]
async fn a_removal_deletes_the_entrys_files_and_the_row_reads_absent() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let models = catalog_served(&repo, "removed", &[("acme-old", 32), ("acme-kept", 32)]);
    let dir = models.dir.clone();
    on_disk(&dir, "acme-old");
    on_disk(&dir, "acme-kept");
    let app = server_on(&loader, "mock-a", models, &config(&repo, &[])).app();
    let (_, listed) = send(&app, Method::GET, "/v1/models", None).await;
    assert!(listed.to_string().contains("acme-old"), "servable before: {listed}");

    let (status, removed) = remove(&app, "acme-old").await;
    assert_eq!(status, StatusCode::OK, "{removed}");
    let files = files_of("acme-old");
    assert_eq!((removed["status"].clone(), removed["model"].clone(), removed["bytes_freed"].clone()), (json!("removed"), json!("acme-old"), json!(files.total())), "{removed}");
    let kept = files_of("acme-kept");
    assert_eq!(files_in(&dir), [kept.artifact.0.clone(), kept.sidecar.0.clone()], "only the other entry's files are left");
    let gone = catalog_row(&app, "acme-old").await;
    assert_eq!((gone["on_disk"].clone(), gone["bytes_on_disk"].clone(), gone["switchable"].clone()), (json!("absent"), json!(0), json!(false)), "{gone}");
    let (_, listed) = send(&app, Method::GET, "/v1/models", None).await;
    assert!(!listed.to_string().contains("acme-old"), "no longer servable: {listed}");
    assert_eq!(listed["data"][0]["id"], "mock-a", "the loaded model serves on: {listed}");
    assert!(repo.seen().is_empty(), "a removal never reaches the endpoint");
    let _ = std::fs::remove_dir_all(dir);
}

/// AC 1, story 27: a partly downloaded entry's removal deletes its partial
/// files too — the finished sidecar and the artifact's `.part` — so Remove
/// also discards an unfinished download.
#[tokio::test]
async fn a_partial_entrys_removal_deletes_its_part_files_too() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let models = catalog_served(&repo, "partial", &[("acme-half", 32)]);
    let dir = models.dir.clone();
    let files = files_of("acme-half");
    std::fs::write(dir.join(&files.sidecar.0), &files.sidecar.1).unwrap();
    std::fs::write(dir.join(format!("{}.part", files.artifact.0)), &files.artifact.1[..4096]).unwrap();
    let app = server_on(&loader, "mock-a", models, &config(&repo, &[])).app();
    assert_eq!(catalog_row(&app, "acme-half").await["on_disk"], "partial");

    let (status, removed) = remove(&app, "acme-half").await;
    assert_eq!((status, removed["bytes_freed"].clone()), (StatusCode::OK, json!(1 + 4096)), "{removed}");
    let mut deleted: Vec<String> = removed["files_removed"].as_array().unwrap().iter().map(|name| name.as_str().unwrap().to_owned()).collect();
    deleted.sort();
    assert_eq!(deleted, [files.sidecar.0.clone(), format!("{}.part", files.artifact.0)], "{removed}");
    assert!(files_in(&dir).is_empty(), "nothing of it is left: {:?}", files_in(&dir));
    let gone = catalog_row(&app, "acme-half").await;
    assert_eq!((gone["on_disk"].clone(), gone["bytes_on_disk"].clone()), (json!("absent"), json!(0)), "{gone}");
    let _ = std::fs::remove_dir_all(dir);
}

/// AC 2: the model loaded is refused with a reason, and nothing of it is
/// deleted — named by its id, or by the file it was loaded from when it is
/// served under another id.
#[tokio::test]
async fn the_model_loaded_is_refused_and_keeps_its_files() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let models = catalog_served(&repo, "loaded", &[("acme-live", 32), ("acme-file", 32)]);
    let dir = models.dir.clone();
    on_disk(&dir, "acme-live");
    on_disk(&dir, "acme-file");
    let app = server_on(&loader, "acme-live", models.clone(), &config(&repo, &[])).app();
    let (status, refused) = remove(&app, "acme-live").await;
    assert_eq!((status, refused["error"]["code"].clone()), (StatusCode::CONFLICT, json!("model_in_use")), "{refused}");
    let message = refused["error"]["message"].as_str().unwrap();
    assert!(message.contains("acme-live") && message.contains("loaded"), "{message}");

    // Served as `custom`, from acme-file's artifact.
    let custom = loader.model("custom").with_source(ignis_server::ModelSource { artifact: dir.join(files_of("acme-file").artifact.0), model: "custom".to_owned() });
    let app = Server::from_active(custom)
        .with_switcher(Switcher::new(Arc::clone(&loader) as _, PATIENT).with_catalog(models, true))
        .with_downloads(Downloads::of(&config(&repo, &[]), None))
        .app();
    let (status, refused) = remove(&app, "acme-file").await;
    assert_eq!((status, refused["error"]["code"].clone()), (StatusCode::CONFLICT, json!("model_in_use")), "{refused}");
    assert!(refused["error"]["message"].as_str().unwrap().contains("custom"), "it names the model loaded from it: {refused}");
    assert_eq!(files_in(&dir).len(), 4, "nothing deleted: {:?}", files_in(&dir));
    let _ = std::fs::remove_dir_all(dir);
}

/// AC 2: the entry being downloaded is refused with a reason while its
/// download runs, its `.part` left to the download; once it lands, it is
/// removed like any other.
#[tokio::test]
async fn the_entry_being_downloaded_is_refused_until_its_download_ends() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let models = catalog_served(&repo, "downloading", &[("acme-new", 32)]);
    let dir = models.dir.clone();
    let files = files_of("acme-new");
    repo.hold(&files.artifact.0, 4096);
    let app = server_on(&loader, "mock-a", models, &config(&repo, &[])).app();
    let (status, _) = send(&app, Method::POST, "/v1/models/download", Some(json!({ "model": "acme-new" }))).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    assert_eq!(catalog_row(&app, "acme-new").await["download"], "running");

    let (status, refused) = remove(&app, "acme-new").await;
    assert_eq!((status, refused["error"]["code"].clone()), (StatusCode::CONFLICT, json!("model_in_use")), "{refused}");
    let message = refused["error"]["message"].as_str().unwrap();
    assert!(message.contains("acme-new") && message.contains("being downloaded"), "{message}");

    repo.release();
    // Its files land a moment before the download says it ended.
    until_row(&app, "acme-new", |row| row["on_disk"] == "complete" && row["download"] == "none").await;
    assert_eq!(files_in(&dir), [files.artifact.0.clone(), files.sidecar.0.clone()], "the download landed whole");
    let (status, removed) = remove(&app, "acme-new").await;
    assert_eq!((status, removed["bytes_freed"].clone()), (StatusCode::OK, json!(files.total())), "{removed}");
    assert!(files_in(&dir).is_empty());
    let _ = std::fs::remove_dir_all(dir);
}

/// AC 2 with GitHub #379: an entry `model download` is writing, which holds
/// its lock, is being downloaded too: the removal is refused with `409
/// download_locked`, and the command's `.part` is left to it. Once the
/// command has ended, the entry is removed, and its `.lock` file stays, as
/// every fetch leaves it, for the next one to lock.
#[tokio::test]
async fn an_entry_model_download_is_writing_is_refused_and_its_lock_file_stays() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let models = catalog_served(&repo, "locked", &[("acme-cli", 32)]);
    let (catalog, dir) = (Arc::clone(&models.catalog), models.dir.clone());
    let files = files_of("acme-cli");
    repo.hold(&files.artifact.0, 4096);
    let app = server_on(&loader, "mock-a", models, &config(&repo, &[])).app();
    let part = dir.join(format!("{}.part", files.artifact.0));

    let command = model_download(&repo, &catalog, &["acme-cli"], &dir);
    let page = async {
        for _ in 0..10_000 {
            if std::fs::metadata(&part).is_ok_and(|meta| meta.len() == 4096) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        let (status, refused) = remove(&app, "acme-cli").await;
        assert_eq!((status, refused["error"]["code"].clone()), (StatusCode::CONFLICT, json!("download_locked")), "{refused}");
        let message = refused["error"]["message"].as_str().unwrap();
        let holder = format!("by `ignis-server model download` (process {})", std::process::id());
        assert!(message.contains("acme-cli is already being downloaded") && message.contains(&holder), "it names the holder: {message}");
        assert_eq!(std::fs::metadata(&part).map(|meta| meta.len()).ok(), Some(4096), "the command's .part is left to it");
        repo.release();
    };
    let ((code, lines), ()) = tokio::join!(command, page);
    assert_eq!(code, 0, "{lines:?}");

    let (status, removed) = remove(&app, "acme-cli").await;
    assert_eq!((status, removed["bytes_freed"].clone()), (StatusCode::OK, json!(files.total())), "{removed}");
    assert!(files_in(&dir).is_empty(), "{:?}", files_in(&dir));
    let lock = ignis_server::download::EntryLock::path(catalog.entry("acme-cli").unwrap(), &dir);
    assert!(lock.is_file(), "the lock file stays: {}", lock.display());
    let _ = std::fs::remove_dir_all(dir);
}


/// AC 2: while a switch loads an entry, its removal is refused — the switch
/// holds every `/v1` route but the ones that report and drive it, `503
/// model_switching` naming it — and once it serves, it is the model loaded.
#[tokio::test]
async fn the_model_a_switch_is_loading_is_refused_and_then_the_model_loaded() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let models = catalog_served(&repo, "switching", &[("acme-next", 32)]);
    let dir = models.dir.clone();
    on_disk(&dir, "acme-next");
    let app = server_on(&loader, "mock-a", models, &config(&repo, &[])).app();
    let release = loader.hold_next_load();
    let (status, _) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "model": "acme-next" }))).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    until_status(&app, "switching").await;

    let (status, refused) = remove(&app, "acme-next").await;
    assert_eq!((status, refused["error"]["code"].clone()), (StatusCode::SERVICE_UNAVAILABLE, json!("model_switching")), "{refused}");
    assert!(refused["error"]["message"].as_str().unwrap().contains("acme-next"), "{refused}");
    release.send(()).expect("the held load waits");
    assert_eq!(until_status(&app, "serving").await["data"][0]["id"], "acme-next");
    let (status, refused) = remove(&app, "acme-next").await;
    assert_eq!((status, refused["error"]["code"].clone()), (StatusCode::CONFLICT, json!("model_in_use")), "{refused}");
    assert_eq!(files_in(&dir).len(), 2, "nothing deleted: {:?}", files_in(&dir));
    let _ = std::fs::remove_dir_all(dir);
}

/// AC 2: a load out of `unloaded` admits every request while it runs (spec
/// model-switch/02), so the route refuses the entry it loads by itself: a
/// request naming another entry loads that one, and its removal meanwhile is
/// `409 model_in_use`, the load saying it is loading.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_model_a_load_out_of_unloaded_is_loading_is_refused() {
    refused_while_a_load_out_of_unloaded_loads("loading", "acme-wake", None).await;
}

/// GitHub #380, story 24: the same, when the model loading is a known model
/// served under another id from the entry's file: the entry is matched by
/// its file, not only by its id.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_entry_a_known_model_loading_out_of_unloaded_loads_from_is_refused() {
    refused_while_a_load_out_of_unloaded_loads("loading-known", "wake-alias", Some("wake-alias")).await;
}

/// A server on `mock-a`, unloaded while idle, then loading `named` out of
/// `unloaded` for a chat, the load held: the removal of `acme-wake` is
/// refused meanwhile, and nothing of it is deleted. `alias` is a known model
/// at `acme-wake`'s artifact.
async fn refused_while_a_load_out_of_unloaded_loads(tag: &str, named: &str, alias: Option<&str>) {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let models = catalog_served(&repo, tag, &[("acme-wake", 32)]);
    let dir = models.dir.clone();
    on_disk(&dir, "acme-wake");
    let known = alias.map(|alias| (alias.to_owned(), dir.join(files_of("acme-wake").artifact.0)));
    let switcher = Switcher::new(Arc::clone(&loader) as _, PATIENT).with_catalog(models, true).with_known_models(known.into_iter().collect());
    let server = Server::from_active(loader.model("mock-a"))
        .with_switcher(switcher)
        .with_downloads(Downloads::of(&config(&repo, &[]), None))
        .with_idle_unload(Duration::from_secs(1));
    let app = server.app();
    let _clock = server.spawn_idle_clock(Duration::from_millis(50)).expect("a clock");
    until_status(&app, "unloaded").await;
    for _ in 0..2_000 {
        if server.switcher.as_ref().unwrap().running().is_none() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    server.update_live(|live| live.idle_unload = Duration::ZERO);

    let release = loader.hold_next_load();
    let chat = {
        let app = app.clone();
        let body = json!({ "model": named, "messages": [{ "role": "user", "content": "hello there" }], "max_tokens": 3 });
        tokio::spawn(async move { send(&app, Method::POST, "/v1/chat/completions", Some(body)).await })
    };
    let loading = until_status(&app, "loading").await;
    assert_eq!(loading["switching"]["to"], named, "{loading}");
    let (status, refused) = remove(&app, "acme-wake").await;
    assert_eq!((status, refused["error"]["code"].clone()), (StatusCode::CONFLICT, json!("model_in_use")), "{refused}");
    let message = refused["error"]["message"].as_str().unwrap();
    assert!(message.contains("acme-wake") && message.contains(named) && message.contains("loading"), "{message}");
    assert_eq!(files_in(&dir).len(), 2, "nothing deleted: {:?}", files_in(&dir));

    release.send(()).expect("the held load waits");
    let (status, completion) = chat.await.unwrap();
    assert_eq!((status, completion["model"].clone()), (StatusCode::OK, json!(named)), "{completion}");
    let _ = std::fs::remove_dir_all(dir);
}

/// An operator catalog of `acme-a` and `acme-b`, which both list
/// `shared.bin` beside a sidecar and an artifact of their own, under a fresh
/// folder. The pins are never checked here: nothing is downloaded.
fn sharing_catalog(tag: &str) -> CatalogModels {
    let dir = repo::fresh_dir(tag);
    let zero = "0".repeat(64);
    let mut yaml = String::from("models:\n");
    for id in ["acme-a", "acme-b"] {
        yaml.push_str(&format!(
            "  - id: {id}\n    repo: acme/{id}\n    revision: v1\n    artifact: {id}.ninfer\n    card_class: 32\n    files:\n      - {{ name: shared.bin, bytes: 3, sha256: \"{zero}\" }}\n      - {{ name: {id}.ninfer.graft.json, bytes: 1, sha256: \"{zero}\" }}\n      - {{ name: {id}.ninfer, bytes: 5, sha256: \"{zero}\" }}\n"
        ));
    }
    let catalog = ignis_server::download::catalog::parse_operator(&yaml, ignis_server::config::file::Format::Yaml, "test").expect("a valid catalog");
    CatalogModels { catalog: Arc::new(catalog), dir }
}

/// AC 3, story 26: a file another entry on disk also lists stays, and the
/// answer names it and that entry, so removing one entry never breaks
/// another; the entry removed then reads partial, holding only that file.
/// Removing the other one too takes the file, since no entry on disk lists
/// it any more.
#[tokio::test]
async fn a_file_another_entry_on_disk_also_lists_stays() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let models = sharing_catalog("shared");
    let dir = models.dir.clone();
    for (name, bytes) in [("shared.bin", 3), ("acme-a.ninfer.graft.json", 1), ("acme-a.ninfer", 5), ("acme-b.ninfer.graft.json", 1), ("acme-b.ninfer", 5)] {
        std::fs::write(dir.join(name), vec![7u8; bytes]).unwrap();
    }
    let app = server_on(&loader, "mock-a", models, &config(&repo, &[])).app();

    let (status, removed) = remove(&app, "acme-a").await;
    assert_eq!(
        (status, removed["bytes_freed"].clone(), removed["files_removed"].clone()),
        (StatusCode::OK, json!(6), json!(["acme-a.ninfer.graft.json", "acme-a.ninfer"])),
        "{removed}"
    );
    assert_eq!(removed["files_kept"], json!([{ "file": "shared.bin", "by": "acme-b" }]), "{removed}");
    assert_eq!(files_in(&dir), ["acme-b.ninfer", "acme-b.ninfer.graft.json", "shared.bin"]);
    assert_eq!(catalog_row(&app, "acme-b").await["on_disk"], "complete", "acme-b is whole");
    assert_eq!(catalog_row(&app, "acme-a").await["on_disk"], "partial", "acme-a holds only the shared file");

    let (status, removed) = remove(&app, "acme-b").await;
    assert_eq!((status, removed["bytes_freed"].clone(), removed["files_kept"].clone()), (StatusCode::OK, json!(9), json!([])), "{removed}");
    assert!(files_in(&dir).is_empty(), "{:?}", files_in(&dir));
    let _ = std::fs::remove_dir_all(dir);
}

/// AC 3 with GitHub #379: a file the entry being downloaded also lists
/// stays too, whatever of that entry is on disk yet — `model download` or
/// the server may be writing it, or have verified it already.
#[tokio::test]
async fn a_file_an_entry_being_downloaded_also_lists_stays() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let models = sharing_catalog("shared-downloading");
    let (catalog, dir) = (Arc::clone(&models.catalog), models.dir.clone());
    for (name, bytes) in [("shared.bin", 3), ("acme-a.ninfer.graft.json", 1), ("acme-a.ninfer", 5)] {
        std::fs::write(dir.join(name), vec![7u8; bytes]).unwrap();
    }
    // A fetch of acme-b holds its lock, as `model download acme-b` does.
    let fetching = ignis_server::download::EntryLock::take(catalog.entry("acme-b").unwrap(), &dir, ignis_server::download::Holder::Command).expect("acme-b is free");
    let app = server_on(&loader, "mock-a", models, &config(&repo, &[])).app();

    let (status, removed) = remove(&app, "acme-a").await;
    assert_eq!((status, removed["files_removed"].clone()), (StatusCode::OK, json!(["acme-a.ninfer.graft.json", "acme-a.ninfer"])), "{removed}");
    assert_eq!(removed["files_kept"], json!([{ "file": "shared.bin", "by": "acme-b" }]), "{removed}");
    assert_eq!(files_in(&dir), ["shared.bin"]);
    drop(fetching);
    let _ = std::fs::remove_dir_all(dir);
}

/// Refused with a reason, deleting nothing: an empty `model`, an id that is
/// no catalog entry (a known model is a path the operator gave, never
/// removed), an entry with nothing under `download.path`, and
/// `download.from_ui` off, which turns the API's removals off with its
/// downloads (ADR 0071). `download.enabled` off does not: a server kept off
/// the network still frees its disk.
#[tokio::test]
async fn a_removal_is_refused_with_a_reason_and_deletes_nothing() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let refused = |app: axum::Router, model: &'static str| async move {
        let (status, body) = remove(&app, model).await;
        (status, body["error"]["code"].as_str().unwrap_or_default().to_owned(), body["error"]["message"].as_str().unwrap_or_default().to_owned())
    };
    let models = catalog_served(&repo, "remove-refused", &[("acme-here", 32), ("acme-none", 32)]);
    let dir = models.dir.clone();
    on_disk(&dir, "acme-here");
    let app = server_on(&loader, "mock-a", models.clone(), &config(&repo, &[])).app();

    let (status, body) = remove(&app, " ").await;
    assert_eq!((status, body["error"]["param"].clone()), (StatusCode::BAD_REQUEST, json!("model")), "{body}");
    let (status, code, message) = refused(app.clone(), "mock-a").await;
    assert_eq!((status, code.as_str()), (StatusCode::NOT_FOUND, "model_not_found"), "{message}");
    assert!(message.contains("not a catalog entry"), "{message}");
    let (status, code, message) = refused(app.clone(), "acme-none").await;
    assert_eq!((status, code.as_str()), (StatusCode::CONFLICT, "model_not_downloaded"), "{message}");
    assert!(message.contains("acme-none") && message.contains("download.path"), "{message}");

    let off = server_on(&loader, "mock-a", models.clone(), &config(&repo, &["--download-from-ui", "false"])).app();
    let (status, code, message) = refused(off, "acme-here").await;
    assert_eq!((status, code.as_str()), (StatusCode::FORBIDDEN, "removal_disabled"), "{message}");
    assert!(message.contains("download.from_ui is false"), "{message}");
    assert_eq!(files_in(&dir).len(), 2, "nothing deleted: {:?}", files_in(&dir));
    // The refusals come before the entry's lock is taken: not even its `.lock` file is created.
    let none_lock = ignis_server::download::EntryLock::path(models.catalog.entry("acme-none").unwrap(), &dir);
    assert!(!none_lock.exists(), "nothing created for an entry not there: {}", none_lock.display());
    let here_lock = ignis_server::download::EntryLock::path(models.catalog.entry("acme-here").unwrap(), &dir);
    assert!(!here_lock.exists(), "nor for one whose removal was refused: {}", here_lock.display());

    let air_gapped = server_on(&loader, "mock-a", models, &config(&repo, &["--download-enabled", "false"])).app();
    let (status, removed) = remove(&air_gapped, "acme-here").await;
    assert_eq!(status, StatusCode::OK, "{removed}");
    assert!(files_in(&dir).is_empty());
    let _ = std::fs::remove_dir_all(dir);
}

/// AC 1: the route takes the same authentication as `PATCH /v1/config` —
/// the server's one API key — and answers the browser's preflight.
#[tokio::test]
async fn the_remove_route_sits_behind_the_api_key_and_answers_the_preflight() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let models = catalog_served(&repo, "remove-keyed", &[("acme-keyed", 32)]);
    let dir = models.dir.clone();
    on_disk(&dir, "acme-keyed");
    let app = server_on(&loader, "mock-a", models, &config(&repo, &[])).with_api_key(ignis_server::config::ApiKey::new("sk-test")).app();
    let keyed = |key: Option<&'static str>| {
        let mut request = Request::builder().method(Method::POST).uri("/v1/models/remove").header(header::CONTENT_TYPE, "application/json");
        if let Some(key) = key {
            request = request.header(header::AUTHORIZATION, format!("Bearer {key}"));
        }
        request.body(Body::from(json!({ "model": "acme-keyed" }).to_string())).unwrap()
    };

    let (status, _) = send(&app, Method::OPTIONS, "/v1/models/remove", None).await;
    assert_eq!(status, StatusCode::OK);
    for key in [None, Some("sk-wrong")] {
        let response = app.clone().oneshot(keyed(key)).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED, "{key:?}");
        assert_eq!(response.headers().get(header::WWW_AUTHENTICATE).unwrap(), "Bearer");
    }
    assert_eq!(files_in(&dir).len(), 2, "nothing deleted without the key");
    let response = app.clone().oneshot(keyed(Some("sk-test"))).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert!(files_in(&dir).is_empty());
    let _ = std::fs::remove_dir_all(dir);
}

/// A removed entry forgets the failure its row kept (GitHub #376, #378): its
/// row reads absent and `download: none`, as one never downloaded.
#[tokio::test]
async fn a_removed_entry_forgets_its_last_failure() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let models = catalog_served(&repo, "remove-failed", &[("acme-short", 32)]);
    let dir = models.dir.clone();
    let files = files_of("acme-short");
    std::fs::write(dir.join(&files.sidecar.0), &files.sidecar.1).unwrap();
    let app = server_on(&loader, "mock-a", models, &config(&repo, &["--download-free-margin", "1000000G"])).app();
    let (status, _) = send(&app, Method::POST, "/v1/models/download", Some(json!({ "model": "acme-short" }))).await;
    assert_eq!(status, StatusCode::INSUFFICIENT_STORAGE);
    assert_eq!(catalog_row(&app, "acme-short").await["download"], "failed");

    let (status, removed) = remove(&app, "acme-short").await;
    assert_eq!(status, StatusCode::OK, "{removed}");
    let row = catalog_row(&app, "acme-short").await;
    assert_eq!((row["on_disk"].clone(), row["download"].clone()), (json!("absent"), json!("none")), "{row}");
    assert!(row.get("download_reason").is_none(), "{row}");
    let _ = std::fs::remove_dir_all(dir);
}

/// AC 4, the server's half: a catalog row names the model the server
/// started on (`start_model`) and the known models (`known_models`) loaded
/// from its files under `download.path`, whatever id each is served under —
/// matched by file, not by id — so a page can say what removing it means
/// before it asks; other rows carry neither. The start model, listed among
/// the known ones as `main` lists it, is named once, as the start model. Once
/// another is loaded it is removed like any entry, and its row still names it.
#[tokio::test]
async fn the_rows_name_the_start_model_and_the_known_models_loaded_from_their_files() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let models = catalog_served(&repo, "start-known", &[("acme-start", 32), ("acme-known", 32), ("acme-plain", 32)]);
    let dir = models.dir.clone();
    for id in ["acme-start", "acme-known", "acme-plain"] {
        on_disk(&dir, id);
    }
    let (start_file, known_file) = (dir.join(files_of("acme-start").artifact.0), dir.join(files_of("acme-known").artifact.0));
    let switcher = Switcher::new(Arc::clone(&loader) as _, PATIENT).with_catalog(models, true).with_known_models(
        [("boot".to_owned(), start_file.clone()), ("acme-known".to_owned(), known_file.clone()), ("acme-twin".to_owned(), known_file)].into(),
    );
    switcher.note_start(Some(&ignis_server::ModelSource { artifact: start_file, model: "boot".to_owned() }));
    let app = Server::from_active(loader.model("mock-a")).with_switcher(switcher).with_downloads(Downloads::of(&config(&repo, &[]), None)).app();

    let start = catalog_row(&app, "acme-start").await;
    assert_eq!((start["start_model"].clone(), start.get("known_models")), (json!("boot"), None), "{start}");
    let known = catalog_row(&app, "acme-known").await;
    assert_eq!((known["known_models"].clone(), known.get("start_model")), (json!(["acme-known", "acme-twin"]), None), "{known}");
    let plain = catalog_row(&app, "acme-plain").await;
    assert!(plain.get("start_model").is_none() && plain.get("known_models").is_none(), "{plain}");

    let (status, removed) = remove(&app, "acme-start").await;
    assert_eq!(status, StatusCode::OK, "{removed}");
    let gone = catalog_row(&app, "acme-start").await;
    assert_eq!((gone["on_disk"].clone(), gone["start_model"].clone()), (json!("absent"), json!("boot")), "{gone}");
    let _ = std::fs::remove_dir_all(dir);
}
