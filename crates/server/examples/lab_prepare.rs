//! Prepare the frozen benchmark without running HTTP or loading GPU weights.
use std::path::Path;
use ignis_artifact::{FrontendSet, Reader};
use ignis_server::artifact_template::ArtifactTemplateProvider;
use serde_json::json;
use sha2::{Digest, Sha256};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 { return Err("usage: lab_prepare ARTIFACT OUTPUT_JSON".into()); }
    let reader = Reader::open(Path::new(&args[1]))?;
    let front = FrontendSet::from_reader(&reader)?;
    // Runtime's Engine uses the primary EOS, not every alternative id in the
    // generation config (the same FrontendSet accessor as server::load).
    let stops=vec![front.eos_token_id().ok_or("missing primary EOS")?];
    let provider = ArtifactTemplateProvider::new(front);
    let names = ["astropy__astropy-12907", "django__django-10097", "sympy__sympy-11618"];
    let mut cases = vec![];
    for (name, source) in names.iter().zip(ignis_bench::coding_agent::FIXTURES) {
        let body = ignis_bench::coding_agent::request(serde_json::from_str(source)?, false);
        let mut case = ignis_server::lab_inputs::prepare(&provider, &body)?;
        case["id"] = json!(name);
        case["fixture_sha256"] = json!(format!("{:x}", Sha256::digest(source.as_bytes())));
        case["stop_ids"] = json!(stops);
        cases.push(case);
    }
    std::fs::write(&args[2], serde_json::to_vec_pretty(&json!({"version":1,"cases":cases}))?)?;
    Ok(())
}
