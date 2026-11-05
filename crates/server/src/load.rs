//! Loading a model (spec model-switch/01): the sequence `main` has always
//! run at start — find, verify and name the artifact, check the start
//! options against it, build its template, then load it on the GPU and start
//! its engine — as two functions the model switch runs too.
//!
//! [`prepare_model`] is everything that can refuse a load short of the GPU;
//! [`load_model`] is the load itself. The split is what lets a switch refuse
//! a bad target while the current model is still serving. Only Qwen3.8-27B
//! artifacts pass preparation. This is startup code, not the per-token path:
//! its logging is the once-per-load kind (`hotpath_lint`
//! watches `runtime.rs`, the scheduler construction it calls into).

use std::sync::Arc;

use ignis_core::compute::ModelFamily;
use ignis_core::{ConcreteScheduler, SchedulerConfig, TokenId};

use crate::runtime::EngineShape;

/// The mock-backed scheduler (ADR 0006, CPU-only): what serves when no
/// artifact is configured, or the binary was not built with `--features
/// cuda` — the entrypoint never silently blocks startup on a missing GPU
/// backend.
pub fn mock_scheduler(model: &str, default_max_tokens: u32) -> Box<dyn ignis_core::Scheduler> {
    let compute: Arc<dyn ignis_core::Compute> = Arc::new(ignis_core::mock::MockCompute::new());
    Box::new(ConcreteScheduler::with_config(
        SchedulerConfig {
            model: model.into(),
            // ADR 0045: the operator's default cap holds on the mock too.
            default_max_tokens,
            ..SchedulerConfig::default()
        },
        compute,
    ))
}

/// The one count of request ids every model this process loads draws from:
/// a switch or a reload builds a new scheduler, and its requests go on from
/// where the last load's stopped, so a `request_id` in the log names one
/// request for the process's life (ADR 0012), not one per load.
fn process_request_ids() -> ignis_core::RequestIds {
    static IDS: std::sync::OnceLock<ignis_core::RequestIds> = std::sync::OnceLock::new();
    IDS.get_or_init(ignis_core::RequestIds::default).clone()
}

/// The engine over a loaded `scheduler`, its request ids drawn from
/// [`process_request_ids`], and its model thread's handle: how every model
/// this process serves text with starts — the start-up load, a switch's, a
/// reload's ([`load_model`]), and the placeholder ([`placeholder_model`]).
fn start_engine(mut scheduler: Box<dyn ignis_core::Scheduler>) -> (crate::engine::Engine, std::thread::JoinHandle<()>) {
    scheduler.share_request_ids(process_request_ids());
    crate::engine::Engine::with_clock_and_driver(scheduler, Arc::new(crate::telemetry::SystemClock))
}

/// What a start with no artifact serves as `model`: the mock compute under
/// the plain template, its engine started as a load's is.
pub fn placeholder_model(model: &str, default_max_tokens: u32) -> crate::ActiveModel {
    let (engine, driver) = start_engine(mock_scheduler(model, default_max_tokens));
    crate::ActiveModel::new(engine, Arc::new(crate::template::SimpleTemplateProvider)).with_driver(driver)
}

/// Why loading a model refused (spec model-switch/01): each variant is one
/// of the refusals the start-up load has always logged under its own event
/// name ([`LoadError::log`], which `main` calls before it exits). The model
/// switch meets the same refusals and does not stop the process: it reports
/// one through its `Display` as the `reason` of `ignis.model.switch_failed`
/// and of `GET /v1/models`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadError {
    /// The artifact path names no file (`ignis.artifact.missing`).
    ArtifactMissing,
    /// No sidecar to verify the artifact against
    /// (`ignis.artifact.sidecar_missing`).
    SidecarMissing(String),
    /// The artifact did not verify, or its identity could not be read
    /// (`ignis.artifact.load_failed`).
    ArtifactInvalid(String),
    /// The start options and the artifact's model disagree
    /// (`ignis.config.model_mismatch`, spec flash-next/04).
    ModelMismatch(String),
    /// A default thinking budget with no close to force
    /// (`ignis.config.thinking_budget_inert`, spec server/08).
    ThinkingBudgetInert(String),
    /// A thinking default the template cannot honour
    /// (`ignis.config.thinking_invalid`, GitHub #68).
    ThinkingInvalid(String),
    /// The artifact's vision processor does not fit the model contract
    /// (`ignis.vision.processor_invalid`, GitHub #179).
    VisionProcessorInvalid(String),
    /// `generation_config.json` carries no `eos_token_id`
    /// (`ignis.model.eos_missing`): a backend that can never stop on EOS
    /// would run every request to its cap.
    EosMissing,
    /// The GPU load itself failed (`ignis.model.load_failed`).
    LoadFailed(String),
    /// The 27B host plan refused the load (ADR
    /// 0062): not enough RAM for its pinned lines and the margin
    /// (`ignis.model.load_failed`; the lines in bytes are
    /// `ignis.runtime.flash_next_host_plan_refused`'s or
    /// `ignis.runtime.qwen27b_host_plan_refused`'s).
    HostPlan(ignis_core::residency::HostPlanError),
}

impl From<crate::runtime::Qwen27bLoadError> for LoadError {
    fn from(error: crate::runtime::Qwen27bLoadError) -> Self {
        match error {
            crate::runtime::Qwen27bLoadError::HostPlan(refusal) => Self::HostPlan(refusal),
            // The VRAM plan's refusal reads as it always did: its deficit and
            // the adaptations tried (ADR 0063 §3).
            crate::runtime::Qwen27bLoadError::VramPlan(refusal) => Self::LoadFailed(refusal.to_string()),
            crate::runtime::Qwen27bLoadError::Other(message) => Self::LoadFailed(message),
        }
    }
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ArtifactMissing => write!(f, "no such file"),
            Self::EosMissing => write!(f, "generation_config.json has no eos_token_id"),
            Self::SidecarMissing(message)
            | Self::ArtifactInvalid(message)
            | Self::ModelMismatch(message)
            | Self::ThinkingBudgetInert(message)
            | Self::ThinkingInvalid(message)
            | Self::VisionProcessorInvalid(message)
            | Self::LoadFailed(message) => write!(f, "{message}"),
            Self::HostPlan(refusal) => write!(f, "{refusal}"),
        }
    }
}

impl LoadError {
    /// Log the refusal of `artifact` under the event name the start-up load
    /// has always used for it, with `outcome` (what happens next: "refusing
    /// to start") as the message. Event names are static in `tracing`, so
    /// each variant names its own.
    pub fn log(&self, artifact: &std::path::Path, outcome: &str) {
        let artifact = artifact.display();
        match self {
            Self::ArtifactMissing => tracing::error!(
                name: "ignis.artifact.missing",
                %artifact,
                "no such file — {outcome}; drop --model-artifact/IGNIS_MODEL_ARTIFACT to fetch the model into --download-path instead"
            ),
            Self::SidecarMissing(error) => {
                tracing::error!(name: "ignis.artifact.sidecar_missing", %artifact, %error, "{outcome}")
            }
            Self::ArtifactInvalid(error) => {
                tracing::error!(name: "ignis.artifact.load_failed", %artifact, %error, "{outcome}")
            }
            Self::ModelMismatch(error) => tracing::error!(name: "ignis.config.model_mismatch", %error, "{outcome}"),
            Self::ThinkingBudgetInert(error) => {
                tracing::error!(name: "ignis.config.thinking_budget_inert", %error, "{outcome}")
            }
            Self::ThinkingInvalid(error) => tracing::error!(name: "ignis.config.thinking_invalid", %error, "{outcome}"),
            Self::VisionProcessorInvalid(error) => {
                tracing::error!(name: "ignis.vision.processor_invalid", %error, "{outcome}")
            }
            Self::EosMissing => tracing::error!(
                name: "ignis.model.eos_missing",
                %artifact,
                "generation_config.json has no eos_token_id — {outcome}"
            ),
            Self::LoadFailed(error) => tracing::error!(name: "ignis.model.load_failed", %artifact, %error, "{outcome}"),
            Self::HostPlan(error) => tracing::error!(name: "ignis.model.load_failed", %artifact, %error, "{outcome}"),
        }
    }
}

/// A model load with everything that can refuse it short of the GPU already
/// done (spec model-switch/01): the artifact found, verified and named, the
/// start options checked against its model, its template, thinking close and
/// vision processor built. [`load_model`] finishes it.
///
/// The split is what lets a model switch refuse a bad target — a wrong path,
/// a checksum that is not clean, flags the target cannot take — while the
/// old model is still serving: only [`load_model`] needs the GPU the old
/// model holds.
pub struct PreparedModel {
    /// The container the load opens.
    pub artifact: std::path::PathBuf,
    /// The id the load is served under ([`crate::config::Config::for_family`]).
    pub model: String,
    /// The artifact's model (ADR 0043).
    pub family: ModelFamily,
    /// The start options the load runs with.
    config: crate::config::Config,
    /// The verified text frontend and optional Vision processor.
    text: PreparedText,
}

/// The template, thinking close and vision processor of a text model's load.
struct PreparedText {
    provider: crate::artifact_template::ArtifactTemplateProvider,
    /// The vision processor, checked against the model contract at the
    /// configured context; [`load_model`] serves it at the context the load
    /// runs ([`serve_vision`]).
    processor: Option<ignis_artifact::vision::VisionProcessor>,
    /// The vision processor's largest item at the configured context, in
    /// tokens: what the encoder's workspace is sized for (a load that adapts
    /// its context lowers it with the envelope, `EngineShape::at`).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    item_bound: Option<u64>,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    thinking_close: Option<Arc<ignis_core::thinking_budget::ThinkingClose>>,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    eos: Option<TokenId>,
}

impl PreparedModel {
    /// The configuration the load runs with, fitted to its family.
    pub fn config(&self) -> &crate::config::Config {
        &self.config
    }
}

/// A loaded model and what its load reserved (`/metrics`, GitHub #216):
/// `None` where nothing was planned (a build without `cuda`).
pub struct LoadedModel {
    /// The model, its engine running, its driver handle kept.
    pub model: crate::ActiveModel,
    /// What the load's VRAM plan reserved.
    pub reservations: Option<crate::metrics::LoadReservations>,
}

/// Everything the start-up load does before the GPU, for `artifact` under
/// `config` (the verified loader path, server-03): the file must exist, its
/// sidecar be present and its checksum report clean; the artifact names its
/// model, which the start options must fit; the thinking defaults must be
/// ones its template honours, and with `--vision-enabled` its processor must match
/// the model contract. Any of those is a [`LoadError`], never a silent
/// fallback to the placeholder.
pub fn prepare_model(config: &crate::config::Config, artifact: &std::path::Path) -> Result<PreparedModel, LoadError> {
    use crate::template::TemplateProvider;

    // A path that is not there at all gets its own refusal (GitHub #234):
    // the sidecar error below would otherwise name a file next to a file
    // that does not exist.
    if !artifact.exists() {
        return Err(LoadError::ArtifactMissing);
    }
    let family = crate::loader::artifact_family(artifact)
        .map_err(|e| LoadError::ArtifactInvalid(e.to_string()))?;
    if family != Some(ModelFamily::Qwen38_27b) {
        return Err(LoadError::ModelMismatch("this fork serves only Qwen3.8-27B".into()));
    }
    let sidecar = crate::loader::find_sidecar(artifact).map_err(|e| LoadError::SidecarMissing(e.to_string()))?;
    let frontend =
        crate::loader::load_artifact(artifact, &sidecar).map_err(|e| LoadError::ArtifactInvalid(e.to_string()))?;
    tracing::info!(
        name: "ignis.artifact.verified",
        artifact = %artifact.display(),
        "checksum clean — tokenizer + chat template loaded"
    );

    // ADR 0043: the artifact names its model, and the start options meet it
    // in one place, which decides the served id or refuses naming the model
    // (spec flash-next/04). An artifact of neither model is served as the
    // 27B always was.
    let family = crate::loader::artifact_family(artifact)
        .map_err(|e| LoadError::ArtifactInvalid(e.to_string()))?
        .unwrap_or(ModelFamily::Qwen38_27b);
    let config = fitted(config, family)?;
    let model = config.model.clone();

    // The thinking budget's forced close (2026-09-24), in this model's own
    // tokens. A tokenizer that splits `</think>` leaves every budget inert:
    // with a default budget configured that is a refusal (spec server/08),
    // since the operator would believe one is active; without one it is said
    // once here rather than discovered per request.
    let thinking_close =
        crate::thinking::thinking_close(|text| frontend.tokenizer().encode(text).map_err(|e| e.to_string()));
    crate::thinking::check_default_budget_close(config.thinking_budget, &thinking_close)
        .map_err(LoadError::ThinkingBudgetInert)?;
    let thinking_close = match thinking_close {
        Ok(close) => Some(Arc::new(close)),
        Err(error) => {
            tracing::warn!(name: "ignis.model.thinking_close_unavailable", %error, "thinking budgets are inert");
            None
        }
    };

    // GitHub #179: a `--vision-enabled` load prepares images with the artifact's
    // processor and acquires them before admission. A tokenizer whose
    // placeholder ids are not the model contract's is refused. Built before
    // the load, which sizes the encoder for the processor's item bound; the
    // load serves it at the context it runs (`serve_vision`, ADR 0063 §3).
    let processor = config
        .vision
        .map(|v| crate::media::load_processor(&frontend, v, config.max_context))
        .transpose()
        .map_err(|e| LoadError::VisionProcessorInvalid(e.to_string()))?;
    // Only the GPU backend needs it; the mock never stops on EOS.
    let eos = frontend.eos_token_id();
    if cfg!(feature = "cuda") && eos.is_none() {
        return Err(LoadError::EosMissing);
    }

    let item_bound = processor.as_ref().map(|p| p.options().max_item_tokens());
    let provider = crate::artifact_template::ArtifactTemplateProvider::new(frontend);

    // A default the template cannot honour is a refusal (a model swap must
    // not silently change behaviour), as a missing EOS token or an unclean
    // checksum is.
    let defaults = crate::thinking::ThinkingDefaults {
        enable_thinking: config.enable_thinking,
        reasoning_effort: config.reasoning_effort,
    };
    crate::thinking::validate_defaults(&defaults, &provider.thinking_capabilities())
        .map_err(LoadError::ThinkingInvalid)?;

    Ok(PreparedModel {
        artifact: artifact.to_path_buf(),
        model,
        family,
        config,
        text: PreparedText { provider, processor, item_bound, thinking_close, eos },
    })
}

/// The options fitted to `family` (spec config-v2/01): at start, its scoped
/// values applied and anything the family cannot take refused; a switch
/// hands over options it already fitted (`config::fit_to_family`, which
/// drops instead of refusing), and those are used as they are.
fn fitted(config: &crate::config::Config, family: ModelFamily) -> Result<crate::config::Config, LoadError> {
    match config.basis.family() {
        Some(fitted) if fitted == family => Ok(config.clone()),
        _ => config.for_family(family).map_err(|e| LoadError::ModelMismatch(e.to_string())),
    }
}

/// Finish a [`PreparedModel`]: build its scheduler — the real GPU-backed one
/// under `cuda`, `MockCompute` over the real template otherwise — and start
/// its engine, keeping the model thread's handle so a switch can join it.
///
/// Blocking for the length of a GPU load (tens of seconds); an async caller
/// runs it on `spawn_blocking`. It needs a tokio runtime context: the
/// engine's telemetry consumer is spawned onto it.
pub fn load_model(prepared: PreparedModel) -> Result<LoadedModel, LoadError> {
    let PreparedModel { artifact, model, family, config, text } = prepared;
    let PreparedText { provider, processor, item_bound, thinking_close, eos } = text;
    #[cfg(feature = "cuda")]
    let (scheduler, reservations, max_context, adapted) = {
        let eos = eos.ok_or(LoadError::EosMissing)?;
        let (scheduler, reserved, max_context, adapted) =
            cuda_scheduler_for(&artifact, &model, family, eos, &config, item_bound, thinking_close)?;
        (scheduler, Some(reserved), max_context, adapted)
    };
    #[cfg(not(feature = "cuda"))]
    let (scheduler, reservations, max_context, adapted) = {
        let _ = (item_bound, thinking_close, eos);
        tracing::warn!(
            name: "ignis.model.mock_compute",
            "built without --features cuda — MockCompute despite --model-artifact/IGNIS_MODEL_ARTIFACT (the templated text is real, the completions are not)"
        );
        (mock_scheduler(&model, config.default_max_tokens), None, config.max_context, String::new())
    };
    let (provider, acquirer) = serve_vision(provider, processor, &config, max_context);
    let (engine, driver) = start_engine(scheduler);
    let mut active = crate::ActiveModel::new(engine, Arc::new(provider))
        .with_driver(driver)
        .with_family(family)
        .with_source(crate::ModelSource { artifact, model })
        .with_adapted(adapted);
    if let Some(acquirer) = acquirer {
        active = active.with_media(acquirer);
    }
    Ok(LoadedModel { model: active, reservations })
}

/// `provider` with the vision processor a `--vision-enabled` load serves,
/// and the acquirer that prepares images with it, both at `max_context`:
/// the context the load runs, which its adaptation may have lowered from
/// the configured one (ADR 0063 §3). So the processor's request budget and
/// item bound are the envelope the leaf planned, never the configured
/// context's.
fn serve_vision(
    provider: crate::artifact_template::ArtifactTemplateProvider,
    processor: Option<ignis_artifact::vision::VisionProcessor>,
    config: &crate::config::Config,
    max_context: u32,
) -> (crate::artifact_template::ArtifactTemplateProvider, Option<Arc<crate::media::MediaAcquirer>>) {
    let Some((processor, vision)) = processor.zip(config.vision) else {
        return (provider, None);
    };
    let processor = crate::media::processor_at(processor, vision, max_context);
    let acquirer = crate::media::MediaAcquirer::new(
        Arc::new(processor.clone()),
        processor.options().clone(),
        crate::media::MediaPolicy::new(config.media.allow_private_network, config.media.cache_bytes),
    );
    (provider.with_vision(processor), Some(Arc::new(acquirer)))
}

/// The shape a text model's GPU load is handed (`cuda_scheduler_for`): the
/// resolution's ([`EngineShape::from`]), its vision encoder
/// sized for the processor's item bound (`vision_item_bound`), and -- spec
/// 27b-16gb/04 AC 5 -- a drafter's head left to `auto` says what it
/// resolved to (`ignis.runtime.draft_head`, ADR 0063), once per load.
pub fn load_shape(config: &crate::config::Config, vision_item_bound: Option<u64>) -> EngineShape {
    let shape = EngineShape::from(config);
    // The encoder holds one item at a time, and the processor bounds an item
    // below the envelope (`ProcessorOptions::max_item_tokens`): the load
    // sizes the encoder's workspace for that.
    let shape = EngineShape {
        vision: shape.vision.map(|vision| match vision_item_bound {
            // Never above the processor's request budget, the u32 envelope.
            Some(bound) => vision.with_item_max_tokens(u32::try_from(bound).expect("an item bound is at most the envelope")),
            None => vision,
        }),
        ..shape
    };
    if let Some(speculation) = shape.speculation.filter(|_| config.settings().spec.draft_head.is_some_and(|head| head.is_auto())) {
        crate::runtime::log_draft_head_auto(speculation.proposal_head());
    }
    shape
}

/// The GPU-backed scheduler for `artifact` (GitHub #61 / P1-25), logged as
/// `ignis.model.loaded` with the shape it runs: the 27B's at the values its
/// adaptation kept (ADR 0063 §3). Beside it, what the load reserved, the
/// context it runs and the adaptations it took, as `ignis.runtime.vram_plan`
/// spells them.
#[cfg(feature = "cuda")]
fn cuda_scheduler_for(
    artifact: &std::path::Path,
    model: &str,
    family: ignis_core::compute::ModelFamily,
    eos: TokenId,
    config: &crate::config::Config,
    vision_item_bound: Option<u64>,
    thinking_close: Option<Arc<ignis_core::thinking_budget::ThinkingClose>>,
) -> Result<(Box<dyn ignis_core::Scheduler>, crate::metrics::LoadReservations, u32, String), LoadError> {
    let shape = load_shape(config, vision_item_bound);
    let loaded = match family {
        ModelFamily::Qwen38_27b => crate::runtime::qwen27b_load(
            artifact,
            model.into(),
            eos,
            shape.with_family_decode_share(ModelFamily::Qwen38_27b),
            thinking_close,
            &config.kv_disk_location,
        )
        .map_err(LoadError::from),
        // Historical artifact family tags remain readable, but cannot run here.
        ModelFamily::FlashNext | ModelFamily::QwenImage21 | ModelFamily::Nemotron3Diarization => Err(LoadError::LoadFailed(format!("{} has no token scheduler", family.name()))),
    };
    // The 27B's context, lanes and head are the ones its plan kept, and so
    // its rope table: what the scheduler admits and `GET /v1/models` says.
    let (scheduler, reserved, shape, adapted) = loaded?;
    tracing::info!(
        name: "ignis.model.loaded",
        artifact = %artifact.display(),
        eos,
        prefill_chunk = shape.prefill_chunk,
        max_context = shape.max_context,
        // GitHub #336: the 27B's are a load setting too.
        decode_lanes = shape.decode_lanes_for(family),
        kv_format = shape.kv_format.as_str(),
        speculation = %shape.speculation.map_or_else(
            || "off".to_owned(),
            |s| format!(
                "{} draft_tokens={} draft_head={}",
                s.backend().as_str(),
                s.draft_tokens(),
                s.proposal_head().as_str()
            )
        ),
        vision = %shape.vision.map_or_else(
            || "off".to_owned(),
            |v| format!("max_tokens={} item_max_tokens={}", v.max_tokens(), v.item_max_tokens())
        ),
        // ADR 0057: the table the load runs, beside the setting it came
        // from (`auto`, `none` or `yarn:F...`, as configured).
        rope_scaling = %shape.rope_scaling,
        rope_scaling_setting = %config.settings().model.rope_scaling,
        "model loaded on the GPU"
    );
    Ok((Box::new(scheduler), reserved, shape.max_context, crate::adaptation::steps_text(&adapted)))
}

#[cfg(all(test, not(feature = "cuda")))]
mod tests {
    use super::*;

    /// A request on `engine`'s model: its id.
    async fn submitted(engine: &crate::engine::Engine) -> ignis_core::RequestId {
        let input = ignis_core::RequestInput {
            decision: None,
            multimodal: None,
            opener_tokens: None,
            user_turn_tokens: None,
            system_block_tokens: None,
            reuse_boundaries: Vec::new(),
            model: "m".into(),
            tokens: vec![1, 2, 3],
            params: ignis_core::DecodeParams { max_tokens: Some(1), ..Default::default() },
            constrained: None,
            forced_literal: None,
            warm_up: false,
        };
        engine.submit(input, ignis_core::RequestClass::Interactive).await.expect("admitted").0
    }

    /// A text model's load prepared over a fixture artifact that holds only
    /// its frontend: on a build without `cuda`, [`load_model`] serves it on
    /// the mock compute, through the path a GPU load takes.
    fn prepared(model: &str) -> (ignis_artifact::fixture::TempArtifact, PreparedModel) {
        use ignis_artifact::fixture::{self, FixtureObject};
        let tokenizer = r#"{"version":"1.0","pre_tokenizer":{"type":"Whitespace"},"model":{"type":"WordLevel","vocab":{"user":0,"unk":1},"unk_token":"unk"}}"#;
        let resources: [(&'static str, &[u8]); 6] = [
            ("frontend/tokenizer.json", tokenizer.as_bytes()),
            ("frontend/tokenizer_config.json", b"{}"),
            ("frontend/chat_template.jinja", b"{%- for m in messages -%}{{ m.role }} {{ m.content }} {%- endfor -%}"),
            ("frontend/generation_config.json", b"{}"),
            ("frontend/preprocessor_config.json", b"{}"),
            ("frontend/video_preprocessor_config.json", b"{}"),
        ];
        let (mut objects, mut payload) = (Vec::new(), Vec::new());
        for (name, bytes) in resources {
            objects.push(FixtureObject::Resource { name, encoding: "raw-bytes-v1", offset: payload.len() as u64, bytes: bytes.len() as u64 });
            payload.extend_from_slice(bytes);
        }
        let artifact = fixture::write_fixture(&objects, &payload, "load-request-ids").expect("fixture");
        let reader = ignis_artifact::Reader::open(&artifact.path).expect("open fixture artifact");
        let provider = crate::artifact_template::ArtifactTemplateProvider::new(ignis_artifact::FrontendSet::from_reader(&reader).expect("frontend set"));
        let crate::config::ConfigOutcome::Config(config) = crate::config::resolve(&[], |_| None).expect("the options resolve") else {
            panic!("a config")
        };
        let text = PreparedText { provider, processor: None, item_bound: None, thinking_close: None, eos: None };
        let prepared = PreparedModel { artifact: artifact.path.clone(), model: model.into(), family: ModelFamily::Qwen38_27b, config, text };
        (artifact, prepared)
    }

    /// Every model the process serves text with starts its engine as a
    /// load's does: the placeholder, then two loads ([`load_model`], the
    /// start-up load's path and a switch's or a reload's), and each one's
    /// requests go on from the one before's ids instead of from 0 again.
    #[tokio::test]
    async fn every_loads_request_ids_go_on_from_the_load_before() {
        let placeholder = placeholder_model("m", 16);
        let mut ids = vec![submitted(placeholder.text_engine()).await];
        drop(placeholder);
        for _ in 0..2 {
            let (_artifact, prepared) = prepared("m");
            let loaded = load_model(prepared).expect("the mock loads");
            ids.push(submitted(loaded.model.text_engine()).await);
        }
        assert!(ids.windows(2).all(|pair| pair[0] < pair[1]), "{ids:?}");
    }

    }
