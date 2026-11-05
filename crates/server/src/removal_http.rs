//! `POST /v1/models/remove` (GitHub #380, spec #374, ADR 0071): a
//! **Removal** of a catalog entry from the models folder — the Server view's
//! Remove — which frees its disk space.
//!
//! It deletes the entry's files under `download.path`, finished or partial,
//! each at its name and its `.part`, and nothing else: a file another entry
//! on disk also lists stays ([`crate::download::command::remove`]). It
//! answers once they are gone, so the entry's `GET /v1/models/catalog` row
//! reads as it now is at the next read.

use std::path::Path;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::api::{error_response, error_response_naming, ApiError};
use crate::download::command::{file_size, lists_file, FileKept};
use crate::download::{CatalogEntry, EntryLock, Holder, NotLocked};
use crate::model_switch::{Slot, Switcher};
use crate::Server;

/// The body of `POST /v1/models/remove`.
#[derive(Deserialize, ToSchema)]
pub(crate) struct RemoveRequest {
    /// The catalog entry to remove, by the id it is served under (`GET
    /// /v1/models/catalog`'s `id`).
    model: String,
}

/// A removal that took place.
#[derive(Serialize, ToSchema)]
pub(crate) struct ModelRemoved {
    /// Always `removed`.
    status: &'static str,
    /// The entry removed.
    model: String,
    /// The bytes the deleted files held: what the removal freed.
    bytes_freed: u64,
    /// The names deleted under `download.path`, a `.part` under its own.
    files_removed: Vec<String>,
    /// The files of the entry left in place, because another entry on disk
    /// or being downloaded also lists them: the entry then reads `partial`.
    files_kept: Vec<FileKept>,
}

/// `POST /v1/models/remove` — delete a catalog entry's files.
#[utoipa::path(
    post,
    path = "/v1/models/remove",
    tag = "models",
    operation_id = "remove_model",
    summary = "Remove a catalog entry's files from the models folder",
    description = "Deletes the files of the catalog entry `model` names (`GET /v1/models/catalog` lists them) under `download.path`, finished or partial: each file at its name and its `.part`, so a partly downloaded entry's unfinished download goes too. Nothing else is touched: not a known model's own path, not the entry's `.lock` file, which stays for the next fetch, and not a file another entry also lists while that entry is on disk (every file of it at its name) or being downloaded; those are named in `files_kept`, and the entry then reads `partial`. Answers once the files are gone, with the bytes freed; `GET /v1/models/catalog` then shows the entry `absent` (or `partial`), `download: none`, its last failure forgotten, and `GET /v1/models` no longer lists it. The entry's lock is held meanwhile, so no fetch of it begins. Logs `ignis.model.removed`.\n\nThe model loaded is refused, whether it serves, was unloaded while idle or is loading out of that, and so is the model a load is loading, the speech model the speech worker has loaded, and the entry being downloaded, by this server or by `ignis-server model download`; each matched by the entry's files as well as its id, so a known model served from them under another id counts. The start model may go: its row names it in `start_model`, and the next start looks for it again. Under the same API key as `PATCH /v1/config` (ADR 0071); `download.from_ui` false refuses it, `download.enabled` false does not.",
    request_body = RemoveRequest,
    responses(
        (status = 200, description = "The entry's files were deleted.", body = ModelRemoved),
        (status = 400, description = "`model` is empty.", body = ApiError),
        (status = 401, description = "The server was started with an API key and the request carried no matching bearer token.", body = ApiError),
        (status = 403, description = "Removing through the API is off (`removal_disabled`): `download.from_ui` is false.", body = ApiError),
        (status = 404, description = "`model` is not a catalog entry (`model_not_found`): a known model is a path the operator gave, and only a catalog entry's files are removed.", body = ApiError),
        (status = 409, description = "The entry is in use (`model_in_use`, the message saying how): the model loaded, served from its files under any id, the model a load is loading, the speech model the speech worker has loaded, or the entry this server is downloading — cancel that first. Or another fetch holds the entry's lock (`download_locked`): `ignis-server model download`, or another server on the folder. Or nothing of it is under `download.path` (`model_not_downloaded`).", body = ApiError),
        (status = 500, description = "A file could not be deleted (`removal_failed`): the message names it, and what was deleted before it.", body = ApiError),
        (status = 501, description = "This server has no catalog to remove models from (`removal_unavailable`).", body = ApiError),
        (status = 503, description = "The loaded model's first traversal has not run yet (`server_not_ready`), or a model switch is under way (`model_switching`): it may be loading this entry.", body = ApiError),
    ),
)]
pub(crate) async fn post_model_remove(State(server): State<Arc<Server>>, Json(req): Json<RemoveRequest>) -> Response {
    if req.model.trim().is_empty() {
        return crate::api::bad_request_param("`model` must name the catalog entry to remove", "model");
    }
    let Some(catalog) = server.switcher.as_deref().and_then(|switcher| switcher.catalog()).cloned() else {
        return error_response(StatusCode::NOT_IMPLEMENTED, "server_error", "removal_unavailable", "this server has no catalog to remove models from");
    };
    // ADR 0071: `download.from_ui` false turns the API's downloads and
    // removals off together. `download.enabled` does not: it keeps the
    // server off the network, and a removal uses none.
    if server.downloads.as_ref().is_some_and(|downloads| !downloads.from_ui()) {
        return error_response(
            StatusCode::FORBIDDEN,
            "invalid_request_error",
            "removal_disabled",
            format!(
                "removing models through the API is off on this server (download.from_ui is false); delete the files of {} under download.path on the server's machine",
                req.model.trim()
            ),
        );
    }
    let Some(entry) = catalog.catalog.entry(&req.model).cloned() else {
        return error_response_naming(
            StatusCode::NOT_FOUND,
            "model_not_found",
            "model_not_found",
            format!("unknown model: {} is not a catalog entry, and only a catalog entry's files are removed; GET /v1/models/catalog lists them", req.model.trim()),
            Some("model"),
        );
    };
    if let Some(reason) = in_use(&server, &entry, &catalog.dir) {
        return error_response_naming(StatusCode::CONFLICT, "invalid_request_error", "model_in_use", reason, Some("model"));
    }
    if crate::download::command::on_disk(&entry, &catalog.dir, &file_size) == crate::download::OnDisk::Absent {
        return error_response_naming(
            StatusCode::CONFLICT,
            "invalid_request_error",
            "model_not_downloaded",
            format!("nothing of {} is under download.path ({}): there is nothing to remove", entry.id, catalog.dir.display()),
            Some("model"),
        );
    }
    let catalog_dir = catalog.dir.clone();
    let removed = tokio::task::spawn_blocking(move || {
        // GitHub #379: the entry's lock, held while its files go, so a fetch
        // of it — `model download`, another server on the folder — is never
        // left writing a `.part` this deletes, and none begins meanwhile.
        let _lock = EntryLock::take(&entry, &catalog.dir, Holder::Server).map_err(|refused| match refused {
            NotLocked::Held(err) => crate::download_http::locked(err),
            NotLocked::Failed(err) => error_response(StatusCode::INTERNAL_SERVER_ERROR, "server_error", "removal_failed", err.to_string()),
        })?;
        let removed = crate::download::command::remove(&entry, &catalog.catalog, &catalog.dir)
            .map_err(|reason| error_response(StatusCode::INTERNAL_SERVER_ERROR, "server_error", "removal_failed", reason))?;
        Ok::<_, Response>((entry, removed))
    })
    .await
    .expect("a removal does not panic");
    let (entry, removed) = match removed {
        Ok(removed) => removed,
        Err(refused) => return refused,
    };
    if let Some(downloads) = &server.downloads {
        downloads.forget(&entry.id);
    }
    tracing::info!(
        name: "ignis.model.removed",
        model = %entry.id,
        repo = %entry.repo,
        dir = %catalog_dir.display(),
        bytes = removed.bytes,
        files = %removed.files.join(","),
        kept = %removed.kept.iter().map(|kept| format!("{} (listed by {})", kept.file, kept.by)).collect::<Vec<_>>().join(","),
        "the entry's files were removed from the models folder; its next download fetches them again"
    );
    (
        StatusCode::OK,
        Json(ModelRemoved { status: "removed", model: entry.id, bytes_freed: removed.bytes, files_removed: removed.files, files_kept: removed.kept }),
    )
        .into_response()
}

/// Why `entry` cannot be removed now, when it cannot: it is the model
/// loaded — by its id, or by the file the loaded model was loaded from,
/// whatever id serves it. Whether it is serving, unloaded while idle (the
/// next request loads it again from that file) or loading out of that. Or
/// the speech model the speech worker has loaded, or the entry being
/// downloaded.
fn in_use(server: &Server, entry: &CatalogEntry, dir: &Path) -> Option<String> {
    // A switch, a reload or a load out of `unloaded` loading it, by its id or
    // from its file under another id (a known model's). The serving gate
    // holds this route during a switch, but a load out of `unloaded` admits
    // every request while it runs.
    let switcher = server.switcher.as_deref();
    if let Some(Slot::Switch(load)) = switcher.and_then(Switcher::running) {
        let from_its_file = || switcher.and_then(|switcher| switcher.resolve(&load.to).ok()).is_some_and(|artifact| lists_file(entry, dir, &artifact));
        if load.to == entry.id || from_its_file() {
            return Some(format!(
                "{} is the model loading now ({}, from {}); removing it would take its files from under the load",
                entry.id, load.to, load.from
            ));
        }
    }
    let active = server.active();
    let loaded = active.model_id();
    if loaded == entry.id || active.source.as_ref().is_some_and(|source| lists_file(entry, dir, &source.artifact)) {
        return Some(format!(
            "{} is the model loaded ({loaded}), which serves from its files; load another model before removing it",
            entry.id
        ));
    }
    if server.downloads.as_ref().and_then(|downloads| downloads.running()).is_some_and(|download| download.entry().id == entry.id) {
        return Some(format!(
            "{} is being downloaded; cancel its download (POST /v1/models/download/cancel), or let it end, before removing it",
            entry.id
        ));
    }
    None
}
