//! Replay output identity: a `.done` marker is meaningful only for the same
//! artifact directory, complete manifest and explicit load shape. CPU-only.

use std::path::Path;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::replay_shape::LoadShape;

pub fn identity(artifact_directory_hash: [u8; 32], manifest: &[u8], shape: LoadShape) -> Value {
    let hex = |bytes: &[u8]| bytes.iter().map(|b| format!("{b:02x}")).collect::<String>();
    json!({
        "schema": "ignis.dflash2-replay-output/1",
        // Reader::content_hash is the existing structural identity, not a
        // weight-payload digest. Do not reread a multi-GB model per replay.
        "artifact_directory_sha256": hex(&artifact_directory_hash),
        "artifact_identity_scope": "Reader::content_hash: structural directory identity, not weight payload SHA256",
        "manifest_sha256": hex(&Sha256::digest(manifest)),
        "shape": {
            "kv_format": shape.kv_format.as_str(), "max_context": shape.max_context,
            "lanes": shape.lanes, "prefill_chunk": shape.prefill_chunk,
            "window": shape.window, "rope": format!("{:?}", shape.rope),
            "vram": format!("{:?}", shape.vram),
        },
        "record_bytes": 9520,
    })
}

/// Refuse mismatched or unidentifiable existing output rather than silently
/// treating another format's completed jobs as this run's output.
pub fn ensure_output_identity(out_dir: &Path, expected: &Value) -> Result<(), String> {
    std::fs::create_dir_all(out_dir).map_err(|e| format!("create output directory: {e}"))?;
    let path = out_dir.join("run-config.json");
    if path.exists() {
        let existing: Value = serde_json::from_slice(
            &std::fs::read(&path).map_err(|e| format!("read replay identity: {e}"))?
        ).map_err(|e| format!("parse replay identity: {e}"))?;
        if existing != *expected {
            return Err("replay output identity differs; use a separate OUT_DIR for each artifact, manifest and load shape".into());
        }
        return Ok(());
    }
    if std::fs::read_dir(out_dir).map_err(|e| format!("list output directory: {e}"))?.next().is_some() {
        return Err("nonempty replay output has no run-config.json identity; preserve it and use a fresh OUT_DIR".into());
    }
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(path)
        .map_err(|e| format!("create replay identity: {e}"))?;
    let bytes = serde_json::to_vec_pretty(expected).map_err(|e| format!("encode replay identity: {e}"))?;
    file.write_all(&bytes).and_then(|_| file.write_all(b"\n"))
        .map_err(|e| format!("write replay identity: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ignis_artifact::Body27b;
    use ignis_core::KvFormat;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);

    struct Temp(std::path::PathBuf);
    impl Temp {
        fn new() -> Self {
            let p = std::env::temp_dir().join(format!("ignis-replay-identity-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
            std::fs::create_dir(&p).unwrap();
            Self(p)
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
    }

    #[test]
    fn a_completed_job_resumes_only_under_the_same_identity() {
        let out = Temp::new();
        let shape = LoadShape::for_body(Body27b::Trellis);
        let config = identity([7; 32], b"manifest", shape);
        ensure_output_identity(&out.0, &config).unwrap();
        std::fs::write(out.0.join("j000.done"), b"done").unwrap();
        ensure_output_identity(&out.0, &config).unwrap();
        for other in [
            identity([8; 32], b"manifest", shape),
            identity([7; 32], b"changed manifest", shape),
            identity([7; 32], b"manifest", LoadShape { kv_format: KvFormat::Nvfp4, ..shape }),
            identity([7; 32], b"manifest", LoadShape { max_context: 139_264, ..shape }),
            identity([7; 32], b"manifest", LoadShape { lanes: 1, ..shape }),
            identity([7; 32], b"manifest", LoadShape { prefill_chunk: 2048, ..shape }),
        ] {
            assert!(ensure_output_identity(&out.0, &other).is_err());
        }
        assert_eq!(std::fs::read(out.0.join("j000.done")).unwrap(), b"done");
    }

    #[test]
    fn legacy_nonempty_or_corrupt_output_is_refused_without_overwriting() {
        let out = Temp::new();
        let done = out.0.join("j000.done");
        std::fs::write(&done, b"old result").unwrap();
        let config = identity([7; 32], b"manifest", LoadShape::for_body(Body27b::Trellis));
        assert!(ensure_output_identity(&out.0, &config).unwrap_err().contains("no run-config"));
        assert!(!out.0.join("run-config.json").exists());
        std::fs::write(out.0.join("run-config.json"), b"incomplete {").unwrap();
        assert!(ensure_output_identity(&out.0, &config).is_err());
        assert_eq!(std::fs::read(done).unwrap(), b"old result");
    }
}
