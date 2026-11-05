//! Server construction over the safe step-ABI compute adapter.
//!
//! [`scheduler`] is the leaf-agnostic construction point: a loader hands it
//! a model handle and EOS token without leaking either into HTTP or
//! scheduler code. [`cuda_scheduler`] (GitHub #61 / P1-25, feature `cuda`)
//! is the production path built on it: materialize an artifact's weights
//! on the device and wrap the loaded model behind
//! [`ignis_runtime::CudaLeaf`] — the binary falls back to CPU `MockCompute`
//! (`main.rs`) only without this feature or without an artifact.

use std::sync::Arc;

use ignis_core::{ConcreteScheduler, SchedulerConfig, TokenId};
use ignis_runtime::{Model, RuntimeCompute, StepLeaf};

/// Build the server's concrete scheduler over a loaded step-ABI model.
pub fn scheduler<L: StepLeaf>(
    config: SchedulerConfig,
    model: Arc<Model<L>>,
    eos: TokenId,
) -> ConcreteScheduler {
    ConcreteScheduler::with_config(config, Arc::new(RuntimeCompute::new(model, eos)))
}

/// The engine shape the operator configured (`--model-prefill-chunk`,
/// `--model-max-context` — GitHub #87, resolved and validated by [`crate::config`]
/// before any loader work starts). Carried as one value so the flags
/// travel together from `main` to the leaf.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineShape {
    /// The prefill chunk width, in tokens (a nonzero multiple of 128).
    pub prefill_chunk: u32,
    /// The decode share, in percent (`--model-decode-share`, GitHub #306); `None`
    /// takes the family's ([`EngineShape::for_family`]), and the 27B's is 0.
    pub decode_share_percent: Option<u32>,
    /// The maximum per-sequence context, in tokens.
    pub max_context: u32,
    /// The generation cap of a request that sends none
    /// (`--model-default-max-tokens`, ADR 0045); 0 = none. A scheduler setting, not
    /// a load option: it never reaches the leaf or a blob's identity.
    pub default_max_tokens: u32,
    /// The KV storage format the load runs on (`--model-kv-format`, GitHub #122).
    pub kv_format: ignis_core::KvFormat,
    /// The KV pool the operator named (`--vram-kv-pool-bytes`), in bytes or
    /// tokens. `None` takes the KV pool policy's size (ADR 0045): the rest of
    /// the VRAM budget on a resident load, 524,288 tokens on an offloaded one.
    pub kv_pool: Option<ignis_core::KvPoolSize>,
    /// How the load's VRAM budget is chosen (GitHub #210, ADR 0030).
    pub vram: ignis_core::VramMode,
    /// The KV-RAM host tier's budget, in bytes (`--reuse-kv-host-pool-bytes`,
    /// P4-07 GitHub #125): pinned host memory for evicted (suspended)
    /// request snapshots, independent of the GPU-resident pool above.
    pub host_pool_bytes: u64,
    /// What a text model's host plan must leave of the physical memory
    /// available at load (`--model-host-margin-bytes`): Flash-Next's, and the
    /// 27B's (ADR 0062).
    pub host_margin_bytes: u64,
    /// Cross-request state reuse (`--reuse-prompt`, GitHub #186, ADR 0029).
    pub prompt_reuse: bool,
    /// Whether the generation opener's page rides the checkpoint capture (ADR
    /// 0029 as amended 2026-10-07, GitHub #306): the family's, set by
    /// [`EngineShape::for_family`]; no flag names it.
    pub opener_page_rides_capture: bool,
    /// The retained slots the load reserves (GitHub #215, #281, ADR 0030):
    /// in VRAM (`--reuse-retained-device`) and in the pinned host block
    /// (`--reuse-retained-host`), both 0 with prompt reuse off unless named.
    pub retained_device_slots: u32,
    pub retained_host_slots: u32,
    /// Whether the operator named the host count; unnamed, a Flash-Next load
    /// takes its family's default ([`EngineShape::for_family`]).
    pub retained_host_named: bool,
    /// How long a retained Interactive checkpoint in KV-RAM keeps its class's
    /// priority (`--reuse-retained-interactive-ttl`, GitHub #190).
    pub retained_interactive_ttl: std::time::Duration,
    /// Speculative decoding (`--spec-backend`/`--spec-draft-tokens`, P5-02 GitHub #150):
    /// `None` binds nothing of the drafter -- on Flash-Next, unless its MTP
    /// head is on by default ([`flash_next_speculation`]).
    pub speculation: Option<ignis_core::Speculation>,
    /// `--spec-backend off` (GitHub #307): no speculation, the default one included.
    pub speculation_off: bool,
    /// The decode lanes (`--spec-decode-lanes`, GitHub #306; the 27B's too
    /// since GitHub #336); 0 = the family's default
    /// ([`EngineShape::decode_lanes_for`]).
    pub decode_lanes: u32,
    /// Vision (`--vision-enabled`/`--vision-max-tokens`, GitHub #177): `None` binds
    /// and reserves nothing of the vision tower.
    pub vision: Option<ignis_core::Vision>,
    /// The text rotary table (`--model-rope-scaling`, GitHub #227):
    /// [`ignis_core::RopeScaling::NONE`] is the linear one, a YaRN factor
    /// rescales the checkpoint's trained position envelope.
    pub rope_scaling: ignis_core::RopeScaling,
    /// The setting [`Self::rope_scaling`] was resolved from (ADR 0057): an
    /// `auto` table is resolved again for a context the load adapts
    /// ([`Self::at`]); a spelled-out one stays.
    pub rope_setting: ignis_core::RopeSetting,
    /// Which of the 27B's context, decode lanes and drafter head its load
    /// may lower when the VRAM plan refuses them (ADR 0063 §3): those the
    /// profile or the hardcoded default gave, as the resolution says
    /// ([`crate::config::Config::adaptable`]). The default shape, which no
    /// resolution made, adapts nothing; Flash-Next's load adapts nothing.
    pub adaptable: crate::adaptation::Adaptable,
    /// The 27B's adaptation ladder: its context step and its spare
    /// reservation (`--model-adapt-context-step`, `--vram-adapt-spare-tokens`,
    /// ADR 0063 §3).
    pub adapt_ladder: crate::adaptation::Ladder,
    /// The 27B's prompt attention precision (`--model-prefill-attention`,
    /// GitHub #335); Flash-Next ignores it.
    pub prefill_attention: ignis_core::PrefillAttention,
    /// How the 27B holds its BF16 exception tensors (`--model-exception-weights`,
    /// GitHub #341): the binder places them converted to FP8 or as stored.
    pub exception_weights: ignis_core::ExceptionWeights,
    /// KV-disk's budget (`--kv-disk-bytes`, spec vram-budget/03); `None`
    /// takes the family's ([`EngineShape::kv_disk_bytes_for`]). Where its
    /// files go is not a shape's: the loader is handed it.
    pub kv_disk_bytes: Option<u64>,
    /// The windows a live move keeps on the link, each way (GitHub #309);
    /// `None` takes the family's ([`ignis_runtime::TransferPace::for_family`]).
    /// No flag names it: the named constants are the policy, and a
    /// measurement harness sets another.
    pub transfer_pace: Option<ignis_runtime::TransferPace>,
}

impl Default for EngineShape {
    /// The configured defaults, taken from [`ignis_runtime`] rather than
    /// restated here — one source of truth for what `ignis-server` runs
    /// with when the operator passes no flags.
    fn default() -> Self {
        Self {
            prefill_chunk: ignis_runtime::DEFAULT_PREFILL_CHUNK,
            decode_share_percent: None,
            max_context: ignis_runtime::DEFAULT_MAX_CONTEXT,
            default_max_tokens: crate::config::DEFAULT_MAX_TOKENS,
            kv_format: ignis_core::KvFormat::default(),
            kv_pool: None,
            vram: ignis_core::VramMode::Derived {
                headroom_bytes: crate::config::DEFAULT_VRAM_HEADROOM_BYTES,
            },
            host_pool_bytes: crate::config::DEFAULT_HOST_POOL_BYTES,
            host_margin_bytes: crate::config::DEFAULT_HOST_MARGIN_BYTES,
            prompt_reuse: crate::config::DEFAULT_PROMPT_REUSE,
            opener_page_rides_capture: false,
            retained_device_slots: crate::config::DEFAULT_RETAINED_DEVICE_SLOTS,
            retained_host_slots: crate::config::DEFAULT_RETAINED_HOST_SLOTS,
            retained_host_named: false,
            retained_interactive_ttl: ignis_core::host::DEFAULT_RETAINED_INTERACTIVE_TTL,
            speculation: None,
            speculation_off: false,
            decode_lanes: 0,
            vision: None,
            rope_scaling: ignis_core::RopeScaling::NONE,
            // A shape built by hand names its table and adapts nothing: a
            // GPU test's shape is refused as it was, never served smaller.
            rope_setting: ignis_core::RopeSetting::Fixed(ignis_core::RopeScaling::NONE),
            adaptable: crate::adaptation::Adaptable::NONE,
            adapt_ladder: crate::adaptation::Ladder::default(),
            prefill_attention: ignis_core::PrefillAttention::default(),
            exception_weights: ignis_core::ExceptionWeights::default(),
            kv_disk_bytes: None,
            transfer_pace: None,
        }
    }
}

impl From<&crate::config::Config> for EngineShape {
    fn from(config: &crate::config::Config) -> Self {
        Self {
            prefill_chunk: config.prefill_chunk,
            decode_share_percent: config.decode_share_percent,
            max_context: config.max_context,
            default_max_tokens: config.default_max_tokens,
            kv_format: config.kv_format,
            kv_pool: config.kv_pool,
            vram: config.vram,
            host_pool_bytes: config.host_pool_bytes,
            host_margin_bytes: config.host_margin_bytes,
            prompt_reuse: config.prompt_reuse,
            opener_page_rides_capture: false,
            retained_device_slots: config.retained_device_slots,
            retained_host_slots: config.retained_host_slots,
            retained_host_named: config.retained_host_named,
            retained_interactive_ttl: std::time::Duration::from_secs(u64::from(
                config.retained_interactive_ttl_secs,
            )),
            speculation: config.speculation,
            speculation_off: config.speculation_off,
            decode_lanes: config.decode_lanes.unwrap_or(0),
            vision: config.vision,
            rope_scaling: config.rope_scaling,
            rope_setting: config.settings().model.rope_scaling,
            adaptable: config.adaptable(),
            adapt_ladder: config.adapt_ladder(),
            prefill_attention: config.prefill_attention,
            exception_weights: config.exception_weights,
            kv_disk_bytes: config.kv_disk_bytes,
            transfer_pace: None,
        }
    }
}

impl EngineShape {
    /// Every retained slot the load reserves, device and host together: the
    /// one pool the scheduler hands out (GitHub #281).
    pub fn retained_slots(&self) -> u32 {
        self.retained_device_slots + self.retained_host_slots
    }

    /// The decode lanes a load of `family` runs (GitHub #336): the operator's
    /// or the profile's, else the family's own -- 8 on the 27B, 3 on
    /// Flash-Next.
    pub fn decode_lanes_for(&self, family: ignis_core::compute::ModelFamily) -> u32 {
        family.decode_lanes(self.decode_lanes)
    }

    /// This shape for an artifact of `family`: the host retained slots the
    /// operator did not name are the family's default when prompt reuse is
    /// on (Flash-Next's 8, spec flash-next/05) -- the config resolved the
    /// 27B's before the artifact named its model. A named count, or reuse
    /// off, is left as it is.
    pub fn for_family(self, family: ignis_core::compute::ModelFamily) -> Self {
        // GitHub #306: and so is an unnamed decode share, and where the
        // opener's page is published.
        let this = Self {
            opener_page_rides_capture: family.opener_page_rides_capture(),
            ..self.with_family_decode_share(family)
        };
        if this.retained_host_named || !this.prompt_reuse {
            return this;
        }
        // The config's own function (ADR 0046), from the lanes as named.
        Self { retained_host_slots: family.default_retained_host_slots(this.decode_lanes), ..this }
    }

    /// The values the 27B's adaptation starts from (ADR 0063 §3): this
    /// shape's context, its lanes and its drafter's head.
    pub fn adaptation_start(&self) -> crate::adaptation::Attempt {
        crate::adaptation::Attempt {
            max_context: self.max_context,
            decode_lanes: self.decode_lanes_for(ignis_core::compute::ModelFamily::Qwen38_27b),
            draft_head: self.speculation.map(|speculation| speculation.proposal_head()),
        }
    }

    /// This 27B shape at `attempt`'s values (ADR 0063 §3), each changed only
    /// where it differs, with what follows it: the context, an `auto` rope
    /// table re-resolved for it (ADR 0057), and the vision encoder's item
    /// bound kept within the envelope at it; the lanes, and the host retained
    /// slots nobody named -- two a lane (GitHub #336), and with them the
    /// plan's per-slot pages; the drafter's head. At its own values, the
    /// shape itself.
    pub fn at(self, attempt: crate::adaptation::Attempt) -> Self {
        let family = ignis_core::compute::ModelFamily::Qwen38_27b;
        let mut shape = self;
        if attempt.max_context != self.max_context {
            shape.max_context = attempt.max_context;
            if self.rope_setting.is_auto() {
                shape.rope_scaling =
                    self.rope_setting.resolve(attempt.max_context).expect("adaptation only lowers a context `auto` resolved");
            }
            // The processor the load serves bounds an item at the envelope
            // of the context it runs (`load::serve_vision`).
            shape.vision = self.vision.map(|vision| {
                let envelope = vision.envelope_tokens(attempt.max_context);
                if vision.item_max_tokens() > envelope { vision.with_item_max_tokens(envelope) } else { vision }
            });
        }
        if attempt.decode_lanes != self.decode_lanes_for(family) {
            shape.decode_lanes = attempt.decode_lanes;
            if self.prompt_reuse && !self.retained_host_named {
                shape.retained_host_slots = family.default_retained_host_slots(attempt.decode_lanes);
            }
        }
        if let (Some(head), Some(speculation)) = (attempt.draft_head, self.speculation) {
            shape.speculation = Some(speculation.with_proposal_head(head));
        }
        shape
    }

    /// KV-disk's budget on a load of `family` (spec vram-budget/03): the
    /// operator's, or the family's -- 4 GiB on Flash-Next, 0 (off) on the
    /// 27B.
    pub fn kv_disk_bytes_for(&self, family: ignis_core::compute::ModelFamily) -> u64 {
        self.kv_disk_bytes.unwrap_or_else(|| ignis_runtime::kv_disk::default_bytes(family))
    }

    /// This shape with the decode share the operator did not name taken from
    /// `family` (GitHub #306): the one step of [`EngineShape::for_family`]
    /// the 27B's load takes too, its retained slots being the config's.
    pub fn with_family_decode_share(self, family: ignis_core::compute::ModelFamily) -> Self {
        let decode_share_percent = Some(self.decode_share_percent.unwrap_or(family.default_decode_share_percent()));
        Self { decode_share_percent, ..self }
    }
}

/// What a load needs to open KV-disk, Tier 2 (spec vram-budget/03).
#[cfg(feature = "cuda")]
struct KvDiskLoad<'a> {
    artifact_path: &'a std::path::Path,
    model_id: &'a str,
    location: &'a ignis_core::ngram_cache::CacheLocation,
    /// The budget the operator or the family named; 0 opens nothing.
    bytes: u64,
    rope_scaling: ignis_core::RopeScaling,
    /// What the tier's IO threads wait on: Flash-Next's prefill gathers.
    gate: ignis_core::ngram_table::GatherGate,
    restore_floor_tokens: u32,
    /// The pace of every live move, through KV-RAM and the disk alike
    /// (GitHub #309).
    pace: ignis_runtime::TransferPace,
}

/// [`scheduler`], with KV-disk when `disk` names a budget and its volume has
/// room above the margin (spec vram-budget/03): the store under the load's
/// identity, its two staging windows pinned now (serving allocates nothing,
/// ADR 0030), and the scheduler's ledger sized to the store's effective
/// budget. A location the tier cannot open, or a volume with no room, leaves
/// the load without the tier and says so: it is a cache, and never refuses a
/// start. The second value is the tier's budget, `None` without it.
#[cfg(feature = "cuda")]
fn scheduler_with_kv_disk<L: StepLeaf>(
    config: SchedulerConfig,
    model: Arc<Model<L>>,
    eos: TokenId,
    disk: KvDiskLoad<'_>,
) -> Result<(ConcreteScheduler, Option<u64>), String> {
    use ignis_runtime::kv_disk::{self, DiskIdentity, DiskStore};

    let compute = RuntimeCompute::new(model, eos).with_transfer_pace(disk.pace);
    let store = if disk.bytes == 0 {
        None
    } else {
        let opened = disk.location.directory(disk.artifact_path, "kv-disk").and_then(|location| {
            let identity = DiskIdentity::of_load(
                disk.model_id,
                &ignis_core::Compute::blob_identity(&compute),
                disk.rope_scaling,
                kv_disk::sidecar_sha256(disk.artifact_path),
            );
            DiskStore::open_on_volume(&location, disk.bytes, identity, disk.gate)
                .map_err(|e| format!("{}: {e}", location.display()))
        });
        match opened {
            Ok(store) => store,
            Err(error) => {
                // hotpath-lint-allow: one line per model load.
                tracing::warn!(
                    name: "ignis.kv_disk.off",
                    %error,
                    "the KV-disk directory cannot be opened: the load goes on without the tier"
                );
                None
            }
        }
    };
    let Some(store) = store else {
        return Ok((ConcreteScheduler::with_config(config, Arc::new(compute)), None));
    };
    let budget_bytes = store.budget_bytes();
    let arena = ignis_core::seq::HostArena::create(kv_disk::STAGING_BYTES)
        .map_err(|e| format!("KV-disk staging: {e}; --kv-disk-bytes 0 turns the tier off"))?;
    let staging = (0..kv_disk::STAGING_WINDOWS)
        .map(|_| {
            arena
                .alloc(kv_disk::WINDOW_BYTES)
                .map(|window| Box::new(window) as ignis_runtime::StagingWindow)
                .map_err(|e| format!("KV-disk staging: {e}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let config = SchedulerConfig {
        kv_disk_capacity_bytes: budget_bytes,
        kv_disk_restore_floor_tokens: disk.restore_floor_tokens,
        ..config
    };
    let compute = compute.with_kv_disk(store, staging);
    Ok((ConcreteScheduler::with_config(config, Arc::new(compute)), Some(budget_bytes)))
}

/// A load's scheduler over `shape`, its pool's pages and its `lanes`
/// (GitHub #336): the decode lanes the leaf's pool has slots for and its
/// model's widest round serves, on both families -- no more requests in
/// flight, in one prefill batch or holding a pool slot than that.
#[cfg(any(feature = "cuda", test))]
fn scheduler_config_for_shape(
    model: String,
    shape: EngineShape,
    kv_page_tokens: u32,
    capacity_pages: u32,
    lanes: u32,
) -> SchedulerConfig {
    SchedulerConfig {
        model,
        max_in_flight: lanes as usize,
        max_prefill_batch: lanes as usize,
        resident_slot_capacity: lanes,
        kv_page_tokens,
        max_sequence_tokens: shape.max_context,
        // ADR 0045: one home for the default cap, on both models' loads.
        default_max_tokens: shape.default_max_tokens,
        kv_capacity_pages: capacity_pages,
        host_capacity_bytes: shape.host_pool_bytes,
        serving_chunk_tokens: shape.prefill_chunk,
        decode_share: f64::from(shape.decode_share_percent.unwrap_or(0)) / 100.0,
        prompt_reuse: shape.prompt_reuse,
        opener_page_rides_capture: shape.opener_page_rides_capture,
        // GitHub #215: the same count the leaf's pool reserved, so every slot
        // the scheduler hands out is one the pool holds -- both kinds, the
        // device ones at the low indices it hands out first (GitHub #281).
        retained_slots: shape.retained_slots(),
        retained_interactive_ttl: shape.retained_interactive_ttl,
        ..SchedulerConfig::default()
    }
}

/// The VRAM plan's lines for a load (GitHub #210), in `ignis_runtime` since
/// GitHub #340 so the replay examples plan with the same lines.
pub use ignis_runtime::vram_lines;

/// A load's retained slots beside its VRAM plan (GitHub #281): how many of
/// each kind, and the pinned host block the host ones live in -- host
/// memory, which is why it rides the plan's event rather than a plan line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetainedSlotsPlan {
    pub device_slots: u32,
    pub host_slots: u32,
    pub host_bytes: u64,
}

/// The startup report of the VRAM plan (GitHub #210): one
/// `ignis.runtime.vram_plan` event with every line in bytes, the mode, what
/// was free, the headroom or the budget, whether the plan oversubscribes and
/// the retained slots of both kinds (GitHub #281) -- then one
/// `ignis.runtime.vram_oversubscribed` warning per reason the start proceeds
/// anyway. `process_held_bytes` is what of the free memory this process
/// already held (GitHub #324, [`ignis_runtime::held_before_load`]).
/// `adapted` is every adaptation the load took to fit the plan (ADR 0063
/// §3), in order: the event's `adapted`, empty when it took none.
/// `vision` is where a vision load keeps its tower (ADR 0075), the event's
/// `vision_residency`; a text load (`None`) logs none.
pub fn log_vram_plan(
    plan: &ignis_core::VramPlan, retained: RetainedSlotsPlan, process_held_bytes: u64,
    adapted: &[crate::adaptation::AdaptStep], vision: Option<ignis_core::VisionResidency>,
) {
    let lines = plan.lines;
    let adapted = crate::adaptation::steps_text(adapted);
    macro_rules! vram_plan_event {
        ($($mode_field:ident = $mode_value:expr),*) => {
            // hotpath-lint-allow: one line per model load.
            tracing::info!(
                name: "ignis.runtime.vram_plan",
                mode = plan.mode.as_str(),
                free_at_start_bytes = plan.free_at_start_bytes,
                process_held_bytes,
                $($mode_field = $mode_value,)*
                budget_bytes = plan.budget_bytes,
                weights_bytes = lines.weights,
                cuda_context_bytes = lines.cuda_context,
                workspace_bytes = lines.workspace,
                trellis_workspace_bytes = lines.trellis_workspace,
                media_embedding_bytes = lines.media_embedding,
                sampling_bytes = lines.sampling,
                decode_graph_bytes = lines.decode_graph,
                verify_round_bytes = lines.verify_round,
                drafter_round_bytes = lines.drafter_round,
                lane_state_bytes = lines.lane_state,
                retained_slots_bytes = lines.retained_slots,
                hq_residual_window_bytes = lines.hq_residual_window,
                residual_bytes = lines.residual,
                kv_pool_bytes = plan.kv_pool_bytes,
                kv_page_count = plan.kv_page_count,
                kv_pool_policy = plan.kv_pool_policy.as_str(),
                kv_pool_pages = plan.kv_page_count,
                kv_pool_tokens = plan.kv_token_capacity(),
                total_bytes = plan.total_bytes,
                allocated_at_load_bytes = plan.total_bytes,
                oversubscribed = plan.oversubscribed,
                retained_device_slots = retained.device_slots,
                retained_host_slots = retained.host_slots,
                retained_host_bytes = retained.host_bytes,
                adapted = %adapted,
                vision_residency = vision.map(ignis_core::VisionResidency::as_str),
                "vram plan"
            )
        };
    }
    match plan.mode {
        ignis_core::VramMode::Derived { headroom_bytes } => {
            vram_plan_event!(headroom_bytes = headroom_bytes)
        }
        ignis_core::VramMode::Explicit {
            allow_oversubscription,
            ..
        } => vram_plan_event!(allow_oversubscription = allow_oversubscription),
    }
    log_vram_warnings(plan);
}

/// `ignis.runtime.draft_head` (ADR 0063, spec 27b-16gb/04 AC 5): a load whose
/// drafter's head is left to `auto` says the head `auto` chose -- the full
/// head on every card class, the shortlist being bound beside it rather than
/// instead of it.
pub fn log_draft_head_auto(head: ignis_core::ProposalHead) {
    // hotpath-lint-allow: one line per model load.
    tracing::info!(
        name: "ignis.runtime.draft_head",
        setting = "auto",
        draft_head = head.as_str(),
        "draft_head auto -> {}",
        head.as_str()
    );
}

/// The 27B's host plan request (ADR 0062): what its load pins for its life
/// -- the token embedding's mapped table (`token_embedding_bytes`), the
/// vision tower's spans beside it under the `host` vision residency
/// (`vision_tower_bytes`, ADR 0075; the two make the plan's mapped
/// allocation), the embedding pool the model's load maps under the same
/// residency (GitHub #415: the shape's [`ignis_core::Vision::pool_bytes`] at
/// its context, the leaf's own sizing), KV-disk's staging windows when the
/// tier is on, the host retained slots and the KV-RAM arena -- against the
/// shape's margin (`model.host_margin_bytes`). No experts and no n-gram table.
pub fn qwen27b_host_request(
    available_physical_bytes: u64,
    token_embedding_bytes: u64,
    vision_tower_bytes: u64,
    retained_host_slots_bytes: u64,
    shape: &EngineShape,
) -> ignis_core::residency::HostPlanRequest {
    let family = ignis_core::compute::ModelFamily::Qwen38_27b;
    let kv_disk_on = shape.kv_disk_bytes_for(family) > 0;
    let vision_pool_bytes = shape
        .vision
        .filter(|vision| vision.residency() == ignis_core::VisionResidency::Host)
        .map_or(0, |vision| vision.pool_bytes(shape.max_context));
    ignis_core::residency::HostPlanRequest {
        available_physical_bytes,
        scope: crate::config::field::scope(family),
        token_embedding_bytes,
        vision_tower_bytes,
        vision_pool_bytes,
        expert_pool_bytes: 0,
        load_staging_bytes: 0,
        ngram_hot_rows_bytes: 0,
        staging_bytes: if kv_disk_on { ignis_runtime::kv_disk::STAGING_BYTES } else { 0 },
        retained_host_slots_bytes,
        kv_ram_arena_bytes: shape.host_pool_bytes,
        margin_bytes: shape.host_margin_bytes,
    }
}

/// A 27B load the host plan refused (ADR 0062), as an agent reads it: one
/// `ignis.runtime.qwen27b_host_plan_refused` event with the 27B's lines, the
/// memory available, the margin, what the plan needed and how far short it
/// fell, in exact bytes, and where a vision load keeps its tower (ADR 0075).
/// Its message is the refusal as a person reads it, the text the load then
/// fails with.
pub fn log_qwen27b_host_plan_refused(
    refusal: &ignis_core::residency::HostPlanError, vision: Option<ignis_core::VisionResidency>,
) {
    let ignis_core::residency::HostPlanError::BelowMargin {
        available_bytes,
        planned_bytes,
        margin_bytes,
        crossing_line,
        lines,
        ..
    } = refusal;
    let line = |name: &str| lines.iter().find(|(line, _)| *line == name).map_or(0, |(_, bytes)| *bytes);
    // hotpath-lint-allow: one line per refused load.
    tracing::warn!(
        name: "ignis.runtime.qwen27b_host_plan_refused",
        available_bytes = *available_bytes,
        planned_bytes = *planned_bytes,
        margin_bytes = *margin_bytes,
        needed_bytes = refusal.needed_bytes(),
        short_bytes = refusal.short_bytes(),
        crossing_line = *crossing_line,
        // Not `token_embedding_bytes`: a key with a bare `token` reads as a
        // credential and is redacted (ignis-logging's record.rs).
        embedding_table_bytes = line("token_embedding"),
        vision_tower_bytes = line("vision_tower"),
        vision_pool_bytes = line("vision_pool"),
        vision_residency = vision.map(ignis_core::VisionResidency::as_str),
        staging_bytes = line("staging"),
        retained_host_slots_bytes = line("retained_host_slots"),
        kv_ram_arena_bytes = line("kv_ram_arena"),
        "{refusal}"
    );
}

/// The 27B's host plan as its load carries it out (ADR 0062): one
/// `ignis.runtime.qwen27b_host_plan` event with every line it pins in bytes,
/// what was available and what the plan leaves, and where a vision load
/// keeps its tower and pool (ADR 0075: `vision_residency`, none on a text
/// load) with the tower's and the pool's own lines.
pub fn log_qwen27b_host_plan(
    request: &ignis_core::residency::HostPlanRequest, host: &ignis_core::residency::HostPlan, shape: &EngineShape,
    vision: Option<ignis_core::VisionResidency>,
) {
    // hotpath-lint-allow: one line per model load.
    tracing::info!(
        name: "ignis.runtime.qwen27b_host_plan",
        available_bytes = request.available_physical_bytes,
        planned_bytes = host.total_bytes,
        left_bytes = host.left_bytes,
        margin_bytes = request.margin_bytes,
        embedding_table_bytes = request.token_embedding_bytes,
        vision_tower_bytes = request.vision_tower_bytes,
        vision_pool_bytes = request.vision_pool_bytes,
        vision_residency = vision.map(ignis_core::VisionResidency::as_str),
        staging_bytes = request.staging_bytes,
        retained_host_slots = shape.retained_host_slots,
        retained_host_bytes = request.retained_host_slots_bytes,
        kv_ram_arena_bytes = request.kv_ram_arena_bytes,
        "qwen3.8-27b host plan"
    );
}

/// ADR 0062: a pin the host plan let through but the host then refused --
/// the plan reads the memory available, and the OS may still decline to lock
/// it. The materialize error, followed by the host plan's own figures, so the
/// refusal reads as the plan's and says what it had counted on. Under the
/// `host` vision residency it names both ways out, as the plan's refusal does
/// (ADR 0075): the device residency, and vision off.
pub fn qwen27b_pin_refused(error: &str, request: &ignis_core::residency::HostPlanRequest, planned_bytes: u64) -> String {
    // ADR 0075: the `host` vision residency's tower is pinned with the table,
    // and its embedding pool by the model's load after (GitHub #415).
    let mut vision_lines = String::new();
    for (name, bytes) in [("vision tower", request.vision_tower_bytes), ("embedding pool", request.vision_pool_bytes)] {
        if bytes > 0 {
            vision_lines.push_str(&format!(", the {name} {bytes} bytes"));
        }
    }
    let ways_out = if vision_lines.is_empty() {
        String::new()
    } else {
        format!(
            "; keep the vision tower and its embedding pool in VRAM {}",
            ignis_core::residency::VISION_WAYS_OUT
        )
    };
    format!(
        "materialize weights: {error}; the host plan had counted {planned_bytes} bytes of pinned host memory \
         (the token embedding {} bytes{vision_lines}) against {} bytes available and a {}-byte margin \
         (--{}-model-host-margin-bytes){ways_out}",
        request.token_embedding_bytes, request.available_physical_bytes, request.margin_bytes, request.scope
    )
}

/// Why a 27B load failed: its host plan's refusal (ADR 0062), kept whole so
/// a switch can report it in bytes as Flash-Next's is; its VRAM plan's
/// refusal after every adaptation it could take (ADR 0063 §3), kept whole so
/// a caller reads the deficit from its fields; or anything else, as the text
/// it always was.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Qwen27bLoadError {
    HostPlan(ignis_core::residency::HostPlanError),
    VramPlan(crate::adaptation::StartRefusal),
    Other(String),
}

impl From<String> for Qwen27bLoadError {
    fn from(message: String) -> Self {
        Self::Other(message)
    }
}

impl std::fmt::Display for Qwen27bLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HostPlan(refusal) => refusal.fmt(f),
            Self::VramPlan(refusal) => refusal.fmt(f),
            Self::Other(message) => f.write_str(message),
        }
    }
}

/// One `ignis.runtime.vram_oversubscribed` warning per reason a plan's start
/// proceeds anyway, on either model's load.
pub fn log_vram_warnings(plan: &ignis_core::VramPlan) {
    for warning in &plan.warnings {
        // hotpath-lint-allow: at most two lines per model load.
        tracing::warn!(name: "ignis.runtime.vram_oversubscribed", warning = %warning, "vram oversubscribed");
    }
}

/// The leaf the operator's [`EngineShape`] is loaded as: the last step
/// before a device exists, and the whole of what `ignis-runtime` is told
/// about the operator's flags.
///
/// Separate from [`cuda_scheduler`] because it is the only link in the chain
/// from a flag to a blob's identity that needs no card — `Config` reaches
/// `EngineShape` by [`From`], `EngineShape` reaches the leaf here, and
/// [`ignis_runtime::CudaLeafConfig::blob_identity`] decides which of *these*
/// fields the identity is made of (GitHub #189). Leaving this inline in
/// `cuda_scheduler` would put a field-by-field copy behind a GPU.
///
/// `kv_pool_bytes` is the pool's payload budget as the VRAM plan resolved it
/// (GitHub #210): the operator's `--vram-kv-pool-bytes`, or the rest of the
/// budget, which only a device can say.
#[cfg(feature = "cuda")]
fn leaf_config_for_shape(shape: EngineShape, kv_pool_bytes: u64) -> ignis_runtime::CudaLeafConfig {
    ignis_runtime::CudaLeafConfig {
        max_context_tokens: shape.max_context,
        kv_format: shape.kv_format,
        kv_pool_bytes,
        // GitHub #336: the pool's slots and the model's widest round.
        slot_count: shape.decode_lanes_for(ignis_core::compute::ModelFamily::Qwen38_27b),
        prefill_chunk_tokens: shape.prefill_chunk,
        speculation: shape.speculation,
        vision: shape.vision,
        rope_scaling: shape.rope_scaling,
        prefill_attention: shape.prefill_attention,
        retained_device_slots: shape.retained_device_slots,
        retained_host_slots: shape.retained_host_slots,
        ..ignis_runtime::CudaLeafConfig::default()
    }
}

/// Build the server's real GPU-backed scheduler (GitHub #61 / P1-25):
/// open a second [`ignis_artifact::Reader`] over `artifact_path` (the
/// caller already verified the container through the loader path — this
/// read is for the device-materialization + model-load path, which needs
/// the reader kept alive, not just the frontend set it returned),
/// materialize the text-scope weights on the device, and wrap the loaded
/// model behind the production [`ignis_runtime::CudaLeaf`].
///
/// GitHub #210 (ADR 0030): before the first large allocation, the load lays
/// every reservation out inside the VRAM budget, logs the plan, and refuses
/// the start when it does not fit; after the load it checks the leaf holds
/// what the plan laid out.
#[cfg(feature = "cuda")]
pub fn cuda_scheduler(
    artifact_path: &std::path::Path,
    model_id: String,
    eos: TokenId,
    shape: EngineShape,
) -> Result<(ConcreteScheduler, crate::metrics::LoadReservations), String> {
    cuda_scheduler_with_thinking_close(artifact_path, model_id, eos, shape, None, &Default::default())
}

/// [`cuda_scheduler`], with the model's close sequence the thinking budget
/// forces (2026-09-24, `ignis_core::thinking_budget`); `None` leaves every
/// budget inert.
#[cfg(feature = "cuda")]
pub fn cuda_scheduler_with_thinking_close(
    artifact_path: &std::path::Path,
    model_id: String,
    eos: TokenId,
    shape: EngineShape,
    thinking_close: Option<Arc<ignis_core::thinking_budget::ThinkingClose>>,
    kv_disk_location: &ignis_core::ngram_cache::CacheLocation,
) -> Result<(ConcreteScheduler, crate::metrics::LoadReservations), String> {
    qwen27b_load(artifact_path, model_id, eos, shape, thinking_close, kv_disk_location)
        .map(|(scheduler, reserved, _, _)| (scheduler, reserved))
        .map_err(|e| e.to_string())
}

/// GitHub #341: what a wider prefill chunk's workspace takes from the KV pool, if anything.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn workspace_saving(workspace: u64, default_workspace: u64) -> Option<u64> {
    (workspace > default_workspace).then(|| workspace - default_workspace)
}

/// The 27B's binder scope for `shape` (GitHub #177, #341): the drafter and
/// the vision tower when asked for, and how the exception tensors are held.
#[cfg(feature = "cuda")]
pub(crate) fn model_scope_27b(shape: &EngineShape) -> ignis_artifact::ModelScope {
    ignis_core::model_load::model_scope(shape.speculation, shape.vision, shape.exception_weights)
}

/// A scope the binder bound, with its plan and handles.
#[cfg(feature = "cuda")]
struct BoundScope {
    scope: ignis_artifact::ModelScope,
    plan: ignis_artifact::MaterializationPlan,
    handles: Vec<ignis_artifact::ObjectHandle>,
}

/// Where `bound` holds `scope`, if it does.
#[cfg(feature = "cuda")]
fn bound_index(bound: &[BoundScope], scope: ignis_artifact::ModelScope) -> Option<usize> {
    bound.iter().position(|bound| bound.scope == scope)
}

/// The binder's plan and handles for `scope`, bound the first time a load
/// asks for it: adaptation plans at a second scope only when it drops the
/// shortlist head (ADR 0063 §3), which is bound beside the full one.
#[cfg(feature = "cuda")]
fn bound_scope<'a>(
    reader: &ignis_artifact::Reader,
    bound: &'a mut Vec<BoundScope>,
    scope: ignis_artifact::ModelScope,
) -> Result<&'a BoundScope, String> {
    let index = match bound_index(bound, scope) {
        Some(index) => index,
        None => {
            let (plan, handles) =
                ignis_artifact::bind_model_scope_27b_with(reader, scope).map_err(|e| format!("bind model scope: {e}"))?;
            bound.push(BoundScope { scope, plan, handles });
            bound.len() - 1
        }
    };
    Ok(&bound[index])
}

/// [`cuda_scheduler_with_thinking_close`], with a host plan's refusal kept
/// whole ([`Qwen27bLoadError::HostPlan`], ADR 0062), and the VRAM plan's
/// ([`Qwen27bLoadError::VramPlan`]): the server's load reads them, and a GPU
/// test reads a plan's deficit from the second. Beside the scheduler and its
/// reservations, the shape the load runs: `shape` at the values its
/// adaptation kept (ADR 0063 §3) -- `shape` itself when the plan fit as
/// configured -- and the adaptations that got there, in order (none when it
/// fit), which `GET /v1/models` reports (#340).
#[cfg(feature = "cuda")]
pub fn qwen27b_load(
    artifact_path: &std::path::Path,
    model_id: String,
    eos: TokenId,
    shape: EngineShape,
    thinking_close: Option<Arc<ignis_core::thinking_budget::ThinkingClose>>,
    kv_disk_location: &ignis_core::ngram_cache::CacheLocation,
) -> Result<(ConcreteScheduler, crate::metrics::LoadReservations, EngineShape, Vec<crate::adaptation::AdaptStep>), Qwen27bLoadError> {
    use crate::adaptation::{AdaptError, Planned};
    use ignis_artifact::{CudaDevice, Reader, materialize};
    use ignis_runtime::{CudaLeaf, KV_PAGE_TOKENS};

    let reader = Reader::open(artifact_path).map_err(|e| format!("open artifact: {e}"))?;
    // GitHub #210: what is free before this process holds anything, as NVML
    // (and so nvidia-smi and Task Manager) reports it, read before the context
    // exists; the context is a plan line. Not `cudaMemGetInfo`, which on
    // Windows ignores the caller's context and reads ~400 MiB optimistic.
    // GitHub #324: on a later load the context exists, and what the process
    // still holds is added back.
    let (nvml_free_bytes, _) = CudaDevice::nvml_memory(0)
        .map_err(|e| format!("VRAM budget: free memory is unreadable: {e}"))?;
    #[cfg(target_os = "linux")]
    let process_held_bytes = CudaDevice::nvml_process_bytes(0)
        .map_err(|e| format!("VRAM budget: process memory is unreadable: {e}"))?;
    #[cfg(not(target_os = "linux"))]
    let process_held_bytes = ignis_runtime::held_before_load(CudaDevice::context_created(), ignis_runtime::qwen27b_loaded());
    let free_at_start_bytes = nvml_free_bytes + process_held_bytes;
    let mut device = CudaDevice::create(0).map_err(|e| format!("CUDA device: {e}"))?;
    let geometry = ignis_core::KvGeometry::qwen38_27b();
    // ADR 0063 §3: the plan at the shape's values, and when it refuses them
    // below its minimum, at each adaptation's in turn -- the leaf's lines
    // planned again at the attempt's context (the hq prompt planes, the rope
    // table), lanes and head, before anything is allocated.
    // P5-02 (GitHub #150) / GitHub #177: the drafter's and the vision tower's
    // objects are bound and uploaded only when the operator asked for them;
    // otherwise the plan is the text scope's alone, as before.
    let mut bound = Vec::new();
    let adapted = crate::adaptation::adapt(shape.adaptation_start(), shape.adaptable, shape.adapt_ladder, |attempt| {
        let shape = shape.at(attempt);
        let BoundScope { plan, handles, .. } = bound_scope(&reader, &mut bound, model_scope_27b(&shape))?;
        let planning = leaf_config_for_shape(shape, 0);
        let reservations = planning.plan_reservations(&reader, plan, handles)?;
        // Asked once where its error can surface: the plan below reads a
        // pool the leaf refuses to plan as one that never fits.
        planning.kv_pool_arena_bytes(1)?;
        let kv_arena_bytes = |pages: u32| planning.kv_pool_arena_bytes(pages).unwrap_or(u64::MAX);
        let planned = ignis_core::plan_vram(&ignis_core::VramRequest {
            mode: shape.vram,
            free_at_start_bytes,
            lines: vram_lines(plan.device_capacity_bytes, reservations.reserved),
            kv_page_bytes: shape.kv_format.page_bytes(geometry),
            max_context_tokens: shape.max_context,
            // Every retained slot keeps one tail page, whichever kind it is.
            retained_slots: shape.retained_slots(),
            kv_pool: shape.kv_pool,
            // ADR 0045: every weight is a line, so the pool takes the rest.
            residency: ignis_core::Residency::Resident,
            // GitHub #243: only so a refusal names the knob the operator set,
            // and (GitHub #415) only where the pool is a VRAM line.
            embedding_pool_named: shape.vision.is_some_and(|v| {
                v.residency() == ignis_core::VisionResidency::Device
                    && v.requested_pool_bytes() != ignis_core::vision::DEFAULT_EMBEDDING_POOL_BYTES
            }),
            kv_arena_bytes: &kv_arena_bytes,
            // Windows WDDM pages an oversubscribed device allocation to system
            // RAM; elsewhere it fails.
            can_page: cfg!(windows),
        });
        Ok::<_, String>(match planned {
            Ok(vram) => Planned::Fits { kv_tokens: vram.kv_token_capacity(), plan: (vram, reservations) },
            Err(refusal) => Planned::Unfit(refusal),
        })
    })
    .map_err(|error| match error {
        AdaptError::Refused(refused) => Qwen27bLoadError::VramPlan(refused),
        AdaptError::Failed(error) => Qwen27bLoadError::Other(error),
    })?;
    let shape = shape.at(adapted.attempt);
    let (vram, reservations) = adapted.plan;
    let index = bound_index(&bound, model_scope_27b(&shape)).expect("the kept plan's scope is bound");
    let BoundScope { plan, handles, .. } = bound.swap_remove(index);
    if shape.prefill_chunk > ignis_runtime::DEFAULT_PREFILL_CHUNK {
        let narrow = EngineShape { prefill_chunk: ignis_runtime::DEFAULT_PREFILL_CHUNK, ..shape };
        let narrow = leaf_config_for_shape(narrow, 0).plan_reservations(&reader, &plan, &handles)?;
        if let Some(bytes) = workspace_saving(reservations.reserved.workspace, narrow.reserved.workspace) {
            // hotpath-lint-allow: one line per model load.
            tracing::info!(
                name: "ignis.runtime.prefill_chunk_hint",
                prefill_chunk = shape.prefill_chunk,
                freed_bytes = bytes,
                freed_tokens = bytes / shape.kv_format.page_bytes(geometry) * 64,
                "--model-prefill-chunk {} frees VRAM for the KV pool", ignis_runtime::DEFAULT_PREFILL_CHUNK
            );
        }
    }
    log_vram_plan(
        &vram,
        RetainedSlotsPlan {
            device_slots: shape.retained_device_slots,
            host_slots: shape.retained_host_slots,
            host_bytes: reservations.retained_host_bytes,
        },
        process_held_bytes,
        &adapted.steps,
        shape.vision.map(|v| v.residency()),
    );
    // ADR 0062: the host plan, measured before the first pinned allocation
    // (the token embedding's, at materialization): the table, KV-disk's
    // staging, the host retained slots and the KV-RAM arena must leave the
    // margin, or the start is refused naming the line that crosses it, as
    // Flash-Next's is. A host whose available memory cannot be read starts
    // unplanned, as Flash-Next does. ADR 0075: under the `host` vision
    // residency the tower shares the table's mapped allocation and the
    // embedding pool takes one of its own (GitHub #415), each on a line of its
    // own, and a start that cannot pin them is refused, never moved to the
    // device.
    let mut host_planned = None;
    if let Some(available) = ignis_core::residency::available_physical_bytes() {
        let tower_bytes = ignis_artifact::vision_host_mapped_bytes(&reader, &plan);
        let request = qwen27b_host_request(
            available,
            plan.host_mapped_capacity_bytes - tower_bytes,
            tower_bytes,
            reservations.retained_host_bytes,
            &shape,
        );
        let vision = shape.vision.map(|v| v.residency());
        let host = ignis_core::residency::plan_host(&request).map_err(|refusal| {
            log_qwen27b_host_plan_refused(&refusal, vision);
            Qwen27bLoadError::HostPlan(refusal)
        })?;
        log_qwen27b_host_plan(&request, &host, &shape, vision);
        host_planned = Some((request, host.total_bytes));
    }

    let artifact = materialize(&reader, &plan, &mut device, None).map_err(|e| match &host_planned {
        Some((request, planned_bytes)) => qwen27b_pin_refused(&e.to_string(), request, *planned_bytes),
        None => format!("materialize weights: {e}"),
    })?;

    let leaf_config = leaf_config_for_shape(shape, vram.kv_budget_bytes(shape.kv_format, geometry));

    // Match the scheduler's KV admission accounting to the pool the leaf
    // actually built. Both sides read the page count from the *same*
    // `kv_pool_plan` call — one byte budget, one format, one derived page
    // count — so growing the configured context (or the pool, or changing
    // the format) can never let admission promise capacity the GPU does not
    // have. The slot count is the load's decode lanes (GitHub #336), and the
    // scheduler below is handed the same number.
    let expected_pages = leaf_config
        .kv_pool_plan()
        .map_err(|e| e.to_string())?
        .page_count;

    // GitHub #213: the whole KV-RAM tier, pinned once here and held for the
    // life of the load — the same figure the scheduler's host tier budgets in
    // bytes, so the ledger and the arena describe one region. A size the host
    // cannot page-lock refuses the start rather than surfacing as a spill
    // that quietly never happens.
    let leaf = CudaLeaf::new(device, reader, artifact, handles, leaf_config)
        .with_kv_ram_arena(shape.host_pool_bytes)?;
    let model = Arc::new(Model::load(Arc::new(leaf)).map_err(|e| format!("model load: {e:?}"))?);
    ignis_runtime::note_qwen27b_loaded();

    // GitHub #98 (P3-02): do not just trust that formula — ask the leaf
    // what it actually built (`ignis_seq_pool_stats`, surfaced through
    // `RuntimeStats::kv_page_count`/`kv_page_bytes`), refuse to start on a
    // disagreement, and hand the admission machine the real, leaf-verified
    // pool's own page count rather than the formula's guess.
    let stats = model.stats().map_err(|e| format!("runtime stats: {e:?}"))?;
    let kv_pool = ignis_core::kv::verified_kv_pool(
        expected_pages,
        ignis_core::kv::LeafPoolGeometry {
            page_count: stats.kv_page_count,
            page_bytes: stats.kv_page_bytes,
        },
    )
    .map_err(|e| e.to_string())?;
    let capacity_pages = kv_pool.block_count() as u32;

    // GitHub #210: the same for every reservation the plan laid out -- the
    // leaf's buffers, read back, must be the plan's lines, or the plan no
    // longer describes what this process holds.
    stats.reserved.check_planned(reservations.reserved, vram.kv_pool_bytes)?;
    // What NVML says the load took, beside what the plan said it would
    // (GitHub #210): every line, the retained slots included -- since GitHub
    // #215 serving allocates nothing more. The delta also moves with anything
    // else on the card.
    if let Ok((free_after_load_bytes, _)) = CudaDevice::nvml_memory(0) {
        // hotpath-lint-allow: one line per model load.
        tracing::info!(
            name: "ignis.runtime.vram_loaded",
            allocated_at_load_bytes = vram.total_bytes,
            free_memory_delta_bytes = free_at_start_bytes.saturating_sub(free_after_load_bytes),
            free_after_load_bytes,
            "vram loaded"
        );
    }

    let family = ignis_core::compute::ModelFamily::Qwen38_27b;
    let kv_disk = KvDiskLoad {
        artifact_path,
        model_id: &model_id,
        location: kv_disk_location,
        bytes: shape.kv_disk_bytes_for(family),
        rope_scaling: shape.rope_scaling,
        gate: Default::default(),
        restore_floor_tokens: ignis_runtime::kv_disk::restore_floor_tokens(family),
        pace: shape.transfer_pace.unwrap_or_else(|| ignis_runtime::TransferPace::for_family(family)),
    };
    // GitHub #336: the scheduler's lanes are the load's, as Flash-Next's are:
    // the pool's slots and the model's widest round.
    let (sched, kv_disk_bytes) = scheduler_with_kv_disk(
        SchedulerConfig {
            thinking_close,
            ..scheduler_config_for_shape(model_id.clone(), shape, KV_PAGE_TOKENS, capacity_pages, leaf_config.slot_count)
        },
        model,
        eos,
        kv_disk,
    )?;
    // GitHub #216 (ADR 0030 §Observability): the plan was a value this
    // function built, read once and dropped. What it reserved outlives it
    // now, so an operator can read it off `/metrics` instead of off the one
    // log line the load wrote. The retained slot count is the scheduler's
    // own, not the flags': prompt reuse off hands out none whatever
    // `--reuse-retained-device` and `--reuse-retained-host` said.
    let reserved = crate::metrics::LoadReservations {
        lines: vram.lines,
        budget_bytes: vram.budget_bytes,
        kv_pool_pages: capacity_pages,
        kv_page_bytes: stats.kv_page_bytes,
        kv_ram_arena_bytes: shape.host_pool_bytes,
        retained_slots: sched.retained_slot_count(),
        retained_host_slots: shape.retained_host_slots,
        retained_host_bytes: reservations.retained_host_bytes,
        kv_disk_bytes,
        };
    Ok((sched, reserved, shape, adapted.steps))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ignis_core::{DecodeParams, RequestClass, RequestInput, Scheduler};

    #[test]
    fn the_default_engine_shape_is_what_the_config_resolves_with_no_flags() {
        // Two ways to say "the defaults" must not drift: the one `main`
        // uses (`config::resolve`) and the one the GPU tests use
        // (`EngineShape::default`).
        let crate::config::ConfigOutcome::Config(config) =
            crate::config::resolve(&["--profile".to_owned(), "none".to_owned()], |_| None).expect("resolve")
        else {
            panic!("expected a runnable config");
        };
        // A resolution says what may adapt and that the rope table is
        // `auto`'s; a shape built by hand adapts nothing (ADR 0063 §3).
        let resolved = EngineShape {
            adaptable: crate::adaptation::Adaptable::ALL,
            rope_setting: ignis_core::RopeSetting::Auto,
            ..EngineShape::default()
        };
        assert_eq!(EngineShape::from(&config), resolved);
    }

    /// Review of #336 part D: the default shape -- every GPU gate's hand-built
    /// one -- adapts nothing, so a gate the card cannot hold is refused as it
    /// was, never served at a context it did not ask for; its rope table is
    /// the one it names.
    #[test]
    fn the_default_shape_adapts_nothing() {
        let shape = EngineShape::default();
        assert_eq!(shape.adaptable, crate::adaptation::Adaptable::NONE);
        assert_eq!(shape.rope_setting, ignis_core::RopeSetting::Fixed(shape.rope_scaling));
        assert_eq!(shape.adapt_ladder, crate::adaptation::Ladder { context_step: 65_536, spare_tokens: 65_536 });
    }

    /// GitHub #335: `--model-prefill-attention` reaches the shape, and from it
    /// the 27B's leaf config the load crosses to the leaf.
    #[test]
    fn the_prefill_attention_flag_reaches_the_shape_and_the_leaf_config() {
        let crate::config::ConfigOutcome::Config(config) = crate::config::resolve(
            &["--profile".to_owned(), "none".to_owned(), "--model-prefill-attention".to_owned(), "fp16".to_owned()],
            |_| None,
        )
        .expect("resolve") else {
            panic!("expected a runnable config");
        };
        let shape = EngineShape::from(&config);
        assert_eq!(shape.prefill_attention, ignis_core::PrefillAttention::Fp16);
        assert_eq!(EngineShape::default().prefill_attention, ignis_core::PrefillAttention::Fp8);
        #[cfg(feature = "cuda")]
        assert_eq!(leaf_config_for_shape(shape, 0).prefill_attention, ignis_core::PrefillAttention::Fp16);
    }

    #[test]
    fn a_wider_chunk_hints_only_when_its_workspace_grows() {
        assert_eq!(workspace_saving(896_464_896, 487_837_184), Some(408_627_712));
        assert_eq!(workspace_saving(908_592_896, 908_592_896), None, "vision's encoder sizes both");
    }

    /// GitHub #341: `--model-exception-weights` reaches the shape, and from it
    /// the binder's scope, which converts the exception tensors only under
    /// `fp8` (the default is `bf16`).
    #[test]
    fn the_exception_weights_flag_reaches_the_shape_and_the_binder_scope() {
        let crate::config::ConfigOutcome::Config(config) = crate::config::resolve(
            &["--profile".to_owned(), "none".to_owned(), "--model-exception-weights".to_owned(), "fp8".to_owned()],
            |_| None,
        )
        .expect("resolve") else {
            panic!("expected a runnable config");
        };
        let shape = EngineShape::from(&config);
        assert_eq!(shape.exception_weights, ignis_core::ExceptionWeights::Fp8);
        assert_eq!(EngineShape::default().exception_weights, ignis_core::ExceptionWeights::Bf16);
        #[cfg(feature = "cuda")]
        {
            assert!(model_scope_27b(&shape).fp8_exceptions);
            assert!(!model_scope_27b(&EngineShape::default()).fp8_exceptions);
        }
    }

    /// The whole chain from an operator's flags to a blob's identity, with no
    /// card in it: `Config` -> `EngineShape` -> `CudaLeafConfig` ->
    /// `BlobIdentity` (GitHub #189). Only the first hop is device-free on its
    /// own, which is why `leaf_config_for_shape` exists.
    #[cfg(feature = "cuda")]
    #[test]
    fn the_operator_flags_reach_the_identity_through_the_load_and_nothing_else() {
        let artifact = ignis_core::ArtifactHash::from_bytes([7; 32]);
        const LAYOUT: u32 = 2;
        let identity_of = |config: &crate::config::Config| {
            let kv_pool_bytes = match config.kv_pool {
                Some(ignis_core::KvPoolSize::Bytes(bytes)) => bytes,
                _ => 1 << 30,
            };
            leaf_config_for_shape(EngineShape::from(config), kv_pool_bytes).blob_identity(artifact, LAYOUT)
        };
        let crate::config::ConfigOutcome::Config(base) =
            crate::config::resolve(&[], |_| None).expect("resolve")
        else {
            panic!("expected a runnable config");
        };

        // Every serving knob at once: the bind address, the timeout, the UI,
        // the metrics listener, the API key, the prefill chunk, the context,
        // both pool budgets, the VRAM budget, `--reuse-prompt` and the
        // retained slots. The KV pool is the one that makes this a rule
        // rather than a nicety: unset, it is the rest of a budget derived from
        // the VRAM free at start, so it differs from one start of the same
        // server to the next, and an identity that moved with it would refuse
        // every blob after a reboot.
        let elsewhere = crate::config::Config {
            bind: "0.0.0.0:9999".into(),
            request_timeout_secs: base.request_timeout_secs + 7,
            ui: !base.ui,
            metrics: Some("127.0.0.1:9101".into()),
            api_key: Some(crate::config::ApiKeySetting::Generate),
            prefill_chunk: 512,
            max_context: base.max_context / 2,
            kv_pool: Some(ignis_core::KvPoolSize::Bytes(8 * 1024 * 1024 * 1024)),
            vram: ignis_core::VramMode::Explicit {
                budget_bytes: 20 * 1024 * 1024 * 1024,
                allow_oversubscription: true,
            },
            host_pool_bytes: base.host_pool_bytes * 2,
            prompt_reuse: !base.prompt_reuse,
            retained_device_slots: 3,
            retained_host_slots: 1,
            ..base.clone()
        };
        assert_ne!(base, elsewhere, "the two configs really do differ");
        assert_eq!(
            identity_of(&base),
            identity_of(&elsewhere),
            "no serving flag reaches what the engine is loaded as"
        );

        // And the two load options that do, end to end from the flag.
        let bf16 = crate::config::Config {
            kv_format: ignis_core::KvFormat::Bf16,
            ..base.clone()
        };
        let hq = crate::config::Config {
            kv_format: ignis_core::KvFormat::HqE8_2b,
            ..base.clone()
        };
        assert_eq!(
            identity_of(&bf16)
                .accepts(&identity_of(&hq))
                .expect_err("another KV format")
                .field,
            ignis_core::IdentityField::KvFormat
        );
        let drafted = crate::config::Config {
            speculation: Some(
                ignis_core::Speculation::new(ignis_core::SpeculativeBackend::Dflash2, 4).unwrap(),
            ),
            ..base.clone()
        };
        assert_eq!(
            identity_of(&base)
                .accepts(&identity_of(&drafted))
                .expect_err("a drafter this load has not bound")
                .field,
            ignis_core::IdentityField::Drafter
        );
    }

    #[test]
    fn the_serving_flags_never_reach_the_load_the_identity_names() {
        // GitHub #189, ADR 0029, upstream half. A blob's compatibility
        // identity is assembled by `CudaLeafConfig::blob_identity` (which
        // tests which of *its* fields count), and `cuda_scheduler` builds
        // that struct out of nothing but an `EngineShape`. So the operator
        // flags below cannot change a blob's identity for a reason stronger
        // than a rule someone remembers: they do not survive the step into
        // the only value the load is configured from.
        let crate::config::ConfigOutcome::Config(base) =
            crate::config::resolve(&[], |_| None).expect("resolve")
        else {
            panic!("expected a runnable config");
        };
        let elsewhere = crate::config::Config {
            bind: "0.0.0.0:9999".into(),
            request_timeout_secs: base.request_timeout_secs + 7,
            ui: !base.ui,
            metrics: Some("127.0.0.1:9101".into()),
            api_key: Some(crate::config::ApiKeySetting::Generate),
            ..base.clone()
        };
        assert_ne!(base, elsewhere, "the two configs really do differ");
        assert_eq!(
            EngineShape::from(&base),
            EngineShape::from(&elsewhere),
            "the bind address, the timeout, the UI, the metrics listener and \
             the API key are not part of what the engine is loaded as"
        );
    }

    #[test]
    fn the_operator_prefill_chunk_reaches_the_scheduler_config() {
        let shape = EngineShape {
            prefill_chunk: 512,
            decode_share_percent: Some(25),
            max_context: 65_536,
            default_max_tokens: 8_192,
            kv_format: ignis_core::KvFormat::Bf16,
            kv_pool: Some(ignis_core::KvPoolSize::Tokens(512 * 1024)),
            vram: EngineShape::default().vram,
            host_pool_bytes: crate::config::DEFAULT_HOST_POOL_BYTES,
            host_margin_bytes: crate::config::DEFAULT_HOST_MARGIN_BYTES,
            prompt_reuse: true,
            opener_page_rides_capture: false,
            retained_device_slots: 2,
            retained_host_slots: 3,
            retained_host_named: true,
            retained_interactive_ttl: std::time::Duration::from_secs(60),
            speculation: None,
            speculation_off: false,
            decode_lanes: 0,
            vision: None,
            rope_scaling: ignis_core::RopeScaling::NONE,
            rope_setting: ignis_core::RopeSetting::Fixed(ignis_core::RopeScaling::NONE),
            adaptable: crate::adaptation::Adaptable::NONE,
            adapt_ladder: crate::adaptation::Ladder::default(),
            prefill_attention: ignis_core::PrefillAttention::default(),
            exception_weights: ignis_core::ExceptionWeights::default(),
            kv_disk_bytes: None,
            transfer_pace: None,
        };

        let config = scheduler_config_for_shape("test-model".into(), shape, 64, 32_768, 8);

        assert_eq!(config.serving_chunk_tokens, 512);
        // ADR 0045: and the default cap.
        assert_eq!(config.default_max_tokens, 8_192);
        // GitHub #306: and the decode share, as a fraction.
        assert_eq!(config.decode_share, 0.25);
        // GitHub #186: the reuse knobs reach the scheduler too -- and the
        // retained slots as the count the leaf's pool reserves (GitHub #215),
        // both kinds in one pool (GitHub #281).
        assert!(config.prompt_reuse);
        assert_eq!(config.retained_slots, 5);
        // GitHub #190: and so does the Interactive TTL.
        assert_eq!(config.retained_interactive_ttl, std::time::Duration::from_secs(60));
    }

    /// The 27B's shape under `rtx5090` with `flags`, as its load is handed it.
    fn rtx5080_27b(flags: &[&str]) -> EngineShape {
        let args: Vec<String> = flags.iter().map(|flag| flag.to_string()).collect();
        let crate::config::ConfigOutcome::Config(config) = crate::config::resolve(&args, |_| None).expect("resolve") else {
            panic!("expected a runnable config");
        };
        EngineShape::from(&config.for_family(ignis_core::compute::ModelFamily::Qwen38_27b).expect("the 27B"))
    }

    /// ADR 0063 §3: the shape carries what may adapt, and at an adapted
    /// attempt re-resolves what follows its values -- an `auto` rope table
    /// for the context (ADR 0057; a spelled-out one stays), the host retained
    /// slots nobody named for the lanes (two a lane), the drafter's head.
    #[test]
    fn a_shape_at_an_adapted_attempt_re_resolves_what_follows_its_values() {
        use crate::adaptation::{Adaptable, Attempt};
        use ignis_core::{ProposalHead, RopeScaling};
        let shape = rtx5080_27b(&[]);
        assert_eq!(shape.adaptable, Adaptable::ALL);
        assert_eq!(rtx5080_27b(&["--model-max-context", "131072"]).adaptable, Adaptable { max_context: false, ..Adaptable::ALL });
        assert_eq!(shape.rope_setting, ignis_core::RopeSetting::Auto);
        assert_eq!(shape.adapt_ladder, crate::adaptation::Ladder { context_step: 65_536, spare_tokens: 65_536 });
        let ladder = rtx5080_27b(&["--model-adapt-context-step", "131072", "--vram-adapt-spare-tokens", "0"]).adapt_ladder;
        assert_eq!(ladder, crate::adaptation::Ladder { context_step: 131_072, spare_tokens: 0 }, "the fields reach the shape");
        assert_eq!((shape.max_context, shape.rope_scaling.factor()), (262_144, 0.0));
        let start = shape.adaptation_start();
        assert_eq!(start, Attempt { max_context: 262_144, decode_lanes: 2, draft_head: Some(ProposalHead::Full) });
        assert_eq!(shape.at(start), shape, "at its own values, itself");

        let narrower = shape.at(Attempt { max_context: 786_432, ..start });
        assert_eq!((narrower.max_context, narrower.rope_scaling.factor()), (786_432, 3.0), "auto re-resolved");
        assert_eq!(shape.at(Attempt { max_context: 196_608, ..start }).rope_scaling, RopeScaling::NONE, "within the trained positions");
        let fixed = rtx5080_27b(&["--model-rope-scaling", "yarn:4"]);
        assert!(!fixed.rope_setting.is_auto());
        assert_eq!(fixed.at(Attempt { max_context: 786_432, ..start }).rope_scaling, fixed.rope_scaling, "a spelled-out table stays");

        // The encoder's item bound stays within the envelope at the context
        // the load runs, as the processor it serves does (`load::serve_vision`).
        let wide = ignis_core::Vision::new(131_072).unwrap().with_item_max_tokens(131_072);
        let seen = EngineShape { vision: Some(wide), ..shape };
        let item = |context| seen.at(Attempt { max_context: context, ..start }).vision.map(|v| v.item_max_tokens());
        assert_eq!(item(65_536), Some(65_536), "the adapted envelope");
        assert_eq!(item(196_608), Some(131_072), "the vision envelope, below the context");
        let unbounded = EngineShape { vision: Some(ignis_core::Vision::default()), ..shape };
        assert_eq!(unbounded.at(Attempt { max_context: 65_536, ..start }).vision, Some(ignis_core::Vision::default()), "none named: the leaf reads the envelope");

        let one = shape.at(Attempt { decode_lanes: 1, ..start });
        assert_eq!((one.decode_lanes_for(ignis_core::compute::ModelFamily::Qwen38_27b), one.retained_host_slots), (1, 2));
        assert_eq!((one.max_context, one.rope_scaling), (shape.max_context, shape.rope_scaling));
        let named = rtx5080_27b(&["--reuse-retained-host", "16"]);
        assert_eq!(named.at(Attempt { decode_lanes: 1, ..start }).retained_host_slots, 16, "a named count stays");
        let off = rtx5080_27b(&["--reuse-prompt", "off"]);
        assert_eq!(off.at(Attempt { decode_lanes: 1, ..start }).retained_host_slots, 0, "reuse off keeps none");

        let full = shape.at(Attempt { draft_head: Some(ProposalHead::Full), ..start });
        assert_eq!(full, shape);
        let short = Attempt { draft_head: Some(ProposalHead::Shortlist), ..start };
        assert_eq!(shape.at(short).speculation.map(|s| s.proposal_head()), Some(ProposalHead::Shortlist));
    }

    #[test]
    fn prompt_reuse_off_reaches_the_scheduler_config() {
        let crate::config::ConfigOutcome::Config(off) =
            crate::config::resolve(&["--reuse-prompt".to_owned(), "off".to_owned()], |_| None)
                .expect("resolve")
        else {
            panic!("expected a runnable config");
        };
        let shape = EngineShape::from(&off);
        let config = scheduler_config_for_shape("test-model".into(), shape, 64, 32_768, 8);
        assert!(!config.prompt_reuse);
        assert_eq!(config.retained_slots, 0, "reuse off reserves no retained slot");
    }

    // ── the VRAM plan (GitHub #210) ──────────────────────────────────────

    #[cfg(feature = "cuda")]
    #[test]
    fn the_leaf_reserves_the_retained_slots_the_scheduler_hands_out() {
        // GitHub #215, #281: the leaf reserves each kind and the scheduler
        // hands out all of them -- a slot the scheduler names is one the
        // pool holds.
        let shape = EngineShape {
            retained_device_slots: 2,
            retained_host_slots: 3,
            ..EngineShape::default()
        };
        let leaf = leaf_config_for_shape(shape, 1 << 30);
        assert_eq!((leaf.retained_device_slots, leaf.retained_host_slots), (2, 3));
        assert_eq!(scheduler_config_for_shape("m".into(), shape, 64, 1024, 8).retained_slots, 5);
    }

    #[test]
    fn the_plan_lines_carry_every_reservation_the_leaf_planned() {
        let reserved = ignis_runtime::ReservedBytes {
            workspace: 1,
            trellis_workspace: 2,
            media_embedding: 3,
            sampling: 4,
            decode_graph: 5,
            verify_round: 6,
            drafter_round: 7,
            lane_state: 8,
            retained_slots: 10,
            hq_residual_window: 11,
            kv_pool: 1 << 40,
        };
        let lines = vram_lines(100, reserved);
        assert_eq!(
            lines.entries().map(|(_, bytes)| bytes),
            [
                100,
                ignis_runtime::CUDA_CONTEXT_BYTES,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                10,
                11,
                ignis_runtime::LOAD_RESIDUAL_BYTES,
            ],
            "every line in plan order, and never the KV pool the plan sizes itself"
        );
    }

    /// One retained image at the serving shape (hq-e8-2b, DFlash2), as the
    /// host block packs it.
    const HOST_IMAGE_BYTES: u64 = 232_532_224;
    /// A default load's retained slots: none in VRAM, the host ones in their
    /// pinned block.
    const DEFAULT_RETAINED: RetainedSlotsPlan = RetainedSlotsPlan {
        device_slots: crate::config::DEFAULT_RETAINED_DEVICE_SLOTS,
        host_slots: crate::config::DEFAULT_RETAINED_HOST_SLOTS,
        host_bytes: crate::config::DEFAULT_RETAINED_HOST_SLOTS as u64 * HOST_IMAGE_BYTES,
    };

    fn captured_vram_plan(plan: &ignis_core::VramPlan) -> Vec<serde_json::Value> {
        captured_adapted_vram_plan(plan, &[])
    }

    fn captured_adapted_vram_plan(plan: &ignis_core::VramPlan, adapted: &[crate::adaptation::AdaptStep]) -> Vec<serde_json::Value> {
        use tracing_subscriber::layer::SubscriberExt;
        let sink = std::sync::Arc::new(ignis_logging::MemorySink::new());
        let subscriber =
            tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone()));
        tracing::subscriber::with_default(subscriber, || log_vram_plan(plan, DEFAULT_RETAINED, PROCESS_HELD, adapted, None));
        sink.lines()
            .iter()
            .map(|line| serde_json::from_str(line).expect("valid json"))
            .collect()
    }

    /// What a reload's plan event names as held (GitHub #324).
    const PROCESS_HELD: u64 = ignis_runtime::CUDA_CONTEXT_BYTES + ignis_runtime::QWEN27B_PROCESS_STATE_BYTES;

    /// ADR 0062: the 27B's host plan holds its token embedding, the host
    /// retained slots and the KV-RAM arena it pins, KV-disk's staging only
    /// with the tier on, against the shape's margin; the refusal names the
    /// 27B's own flags.
    #[test]
    fn the_27b_host_plan_holds_its_table_its_reuse_lines_and_kv_disks_staging_when_on() {
        const TABLE: u64 = 1_350_860_800;
        let shape = EngineShape {
            host_pool_bytes: 8 << 30,
            host_margin_bytes: 3 << 30,
            ..EngineShape::default()
        };
        let request = qwen27b_host_request(64 << 30, TABLE, 0, 2_080_000_000, &shape);
        assert_eq!(request.scope, "qwen38");
        assert_eq!(request.token_embedding_bytes, TABLE);
        assert_eq!(request.retained_host_slots_bytes, 2_080_000_000);
        assert_eq!(request.kv_ram_arena_bytes, 8 << 30);
        assert_eq!(request.margin_bytes, 3 << 30);
        assert_eq!(request.staging_bytes, 0, "KV-disk is off on the 27B unless named");
        assert_eq!(
            (request.expert_pool_bytes, request.load_staging_bytes, request.ngram_hot_rows_bytes),
            (0, 0, 0),
            "no experts and no n-gram table"
        );
        let with_disk = EngineShape { kv_disk_bytes: Some(4 << 30), ..shape };
        assert_eq!(
            qwen27b_host_request(64 << 30, TABLE, 0, 0, &with_disk).staging_bytes,
            ignis_runtime::kv_disk::STAGING_BYTES
        );

        let refusal = ignis_core::residency::plan_host(&qwen27b_host_request(1 << 30, TABLE, 0, 0, &shape))
            .expect_err("a GiB cannot hold the table beside the margin");
        assert!(refusal.to_string().contains("(--qwen38-model-host-margin-bytes)"), "{refusal}");

        // ADR 0075 (GitHub #414): the `host` residency's tower, on its own line.
        const TOWER: u64 = 297_000_960;
        let request = qwen27b_host_request(64 << 30, TABLE, TOWER, 0, &shape);
        assert_eq!((request.token_embedding_bytes, request.vision_tower_bytes), (TABLE, TOWER));
        let refusal = ignis_core::residency::plan_host(&qwen27b_host_request((3 << 30) + TABLE + (100 << 20), TABLE, TOWER, 0, &shape))
            .expect_err("the table fits beside the 3 GiB margin, the tower does not");
        let ignis_core::residency::HostPlanError::BelowMargin { crossing_line, .. } = &refusal;
        assert_eq!(*crossing_line, "vision_tower", "{refusal}");
        assert!(refusal.to_string().contains("--vision-residency device"), "{refusal}");

        // GitHub #415: and its embedding pool, host-mapped, on a line of its
        // own: the pool as the leaf sizes it, one envelope-wide item of 5,120
        // BF16 rows unless named, at the shape's context.
        use ignis_core::VisionResidency;
        let at = |residency: VisionResidency, vision: ignis_core::Vision| EngineShape {
            vision: Some(vision.with_residency(residency)),
            max_context: 262_144,
            ..shape
        };
        let envelope = ignis_core::Vision::new(4096).unwrap();
        let request = qwen27b_host_request(64 << 30, TABLE, TOWER, 0, &at(VisionResidency::Host, envelope));
        assert_eq!(request.vision_pool_bytes, 4096 * 5120 * 2, "one 4,096-token item, 32 column pages");
        let named = at(VisionResidency::Host, envelope.with_pool_bytes(640 << 20));
        assert_eq!(qwen27b_host_request(64 << 30, TABLE, TOWER, 0, &named).vision_pool_bytes, 640 << 20);
        let device = at(VisionResidency::Device, envelope.with_pool_bytes(640 << 20));
        assert_eq!(qwen27b_host_request(64 << 30, TABLE, 0, 0, &device).vision_pool_bytes, 0, "a VRAM line under device");
        assert_eq!(qwen27b_host_request(64 << 30, TABLE, 0, 0, &shape).vision_pool_bytes, 0, "none without vision");
        // The table and the tower fit beside the margin, the pool does not.
        let refusal = ignis_core::residency::plan_host(&qwen27b_host_request(
            (3 << 30) + TABLE + TOWER + (10 << 20),
            TABLE,
            TOWER,
            0,
            &at(VisionResidency::Host, envelope),
        ))
        .expect_err("the pool does not fit");
        let ignis_core::residency::HostPlanError::BelowMargin { crossing_line, .. } = &refusal;
        assert_eq!(*crossing_line, "vision_pool", "{refusal}");
        assert_eq!(refusal.short_bytes(), 4096 * 5120 * 2 - (10 << 20) + (8 << 30), "the pool's bytes and the arena's after it");
        assert!(refusal.to_string().contains("--vision-residency device"), "{refusal}");
    }

    /// GitHub #415: `--vision-embedding-pool-mib` sizes the host-mapped pool
    /// under `host` with the default and the floor it has on the device: one
    /// envelope-wide item unless named, and a named pool below it raised to it.
    #[test]
    fn the_embedding_pool_mib_sizes_the_host_plans_pool_line_under_host() {
        let pool = |flags: &[&str]| {
            qwen27b_host_request(64 << 30, 1_350_860_800, 297_000_960, 0, &rtx5080_27b(flags)).vision_pool_bytes
        };
        let host = ["--vision-enabled", "--vision-max-tokens", "4096", "--vision-residency", "host"];
        assert_eq!(pool(&host), 4096 * 5120 * 2, "the default: one envelope-wide item");
        assert_eq!(pool(&[&host[..], &["--vision-embedding-pool-mib", "640"]].concat()), 640 << 20);
        assert_eq!(pool(&[&host[..], &["--vision-embedding-pool-mib", "1"]].concat()), 4096 * 5120 * 2, "raised to the floor");
        let device = ["--vision-enabled", "--vision-max-tokens", "4096", "--vision-residency", "device", "--vision-embedding-pool-mib", "640"];
        assert_eq!(pool(&device), 0, "the device's pool is a VRAM line");
        let text_only = qwen27b_host_request(64 << 30, 1_350_860_800, 0, 0, &{
            let mut shape = rtx5080_27b(&host);
            shape.vision = None;
            shape
        });
        assert_eq!(text_only.vision_pool_bytes, 0, "no vision, no pool");
    }

    /// A pin the host refuses after its plan passed (the OS may decline to lock
    /// memory it reports available) fails with the plan's own figures and flag
    /// beside the materialize error, not the bare error.
    #[test]
    fn a_pin_the_host_refuses_after_the_plan_passed_reads_with_the_plans_figures() {
        let shape = EngineShape { host_pool_bytes: 8 << 30, ..EngineShape::default() };
        let request = qwen27b_host_request(64 << 30, 1_350_864_896, 0, 0, &shape);
        let host = ignis_core::residency::plan_host(&request).expect("64 GiB holds the plan");
        let message = qwen27b_pin_refused("the host cannot pin that much memory", &request, host.total_bytes);
        assert!(message.starts_with("materialize weights: the host cannot pin that much memory; "), "{message}");
        assert!(message.contains(&format!("{} bytes of pinned host memory", host.total_bytes)), "{message}");
        assert!(message.contains("the token embedding 1350864896 bytes"), "{message}");
        assert!(message.contains(&format!("{} bytes available", 64u64 << 30)), "{message}");
        assert!(message.contains("(--qwen38-model-host-margin-bytes)"), "{message}");
        assert!(!message.contains("vision tower"), "no tower, no word of one: {message}");
        // ADR 0075: a `host` start's pin counted the tower too, and says so.
        let request = qwen27b_host_request(64 << 30, 1_350_864_896, 297_000_960, 0, &shape);
        let host = ignis_core::residency::plan_host(&request).expect("64 GiB holds the plan");
        let message = qwen27b_pin_refused("the host cannot pin that much memory", &request, host.total_bytes);
        assert!(message.contains("the token embedding 1350864896 bytes, the vision tower 297000960 bytes)"), "{message}");
        // GitHub #415: and the embedding pool its model load pins after.
        let hosted = EngineShape { vision: Some(host_vision(4096)), max_context: 262_144, ..shape };
        let request = qwen27b_host_request(64 << 30, 1_350_864_896, 297_000_960, 0, &hosted);
        let host = ignis_core::residency::plan_host(&request).expect("64 GiB holds the plan");
        let message = qwen27b_pin_refused("the host cannot pin that much memory", &request, host.total_bytes);
        assert!(
            message.contains(
                "the token embedding 1350864896 bytes, the vision tower 297000960 bytes, the embedding pool 41943040 bytes)"
            ),
            "{message}"
        );
    }

    /// ADR 0075: a `host` start whose pin the host refuses after the plan
    /// passed names both ways out, as the plan's refusal and the leaf's do:
    /// the device residency and vision off. A pin with no vision in it names
    /// neither.
    #[test]
    fn a_pin_the_host_refuses_under_the_host_residency_names_both_ways_out() {
        const WAYS_OUT: &str = "keep the vision tower and its embedding pool in VRAM \
                                (vision.residency: device, --vision-residency device), turn vision off";
        let shape = EngineShape { host_pool_bytes: 8 << 30, ..EngineShape::default() };
        let hosted = EngineShape { vision: Some(host_vision(4096)), max_context: 262_144, ..shape };
        for (label, tower, shape) in [("tower", 297_000_960, &shape), ("tower and pool", 297_000_960, &hosted)] {
            let request = qwen27b_host_request(64 << 30, 1_350_864_896, tower, 0, shape);
            let host = ignis_core::residency::plan_host(&request).expect("64 GiB holds the plan");
            let message = qwen27b_pin_refused("the host cannot pin that much memory", &request, host.total_bytes);
            assert!(message.contains(WAYS_OUT), "{label}: {message}");
        }
        let request = qwen27b_host_request(64 << 30, 1_350_864_896, 0, 0, &shape);
        let host = ignis_core::residency::plan_host(&request).expect("64 GiB holds the plan");
        let message = qwen27b_pin_refused("the host cannot pin that much memory", &request, host.total_bytes);
        assert!(!message.contains("--vision-residency"), "no vision, no way out of it: {message}");
    }

    /// A vision envelope of `tokens` under the `host` residency.
    fn host_vision(tokens: u32) -> ignis_core::Vision {
        ignis_core::Vision::new(tokens).unwrap().with_residency(ignis_core::VisionResidency::Host)
    }

    /// The one event [`log_qwen27b_host_plan_refused`] logs: the 27B's lines
    /// in bytes beside the person's message.
    #[test]
    fn a_refused_27b_host_plan_logs_its_lines_in_bytes_beside_the_persons_message() {
        use tracing_subscriber::layer::SubscriberExt;
        let shape = EngineShape {
            host_pool_bytes: 8 << 30,
            vision: Some(host_vision(4096)),
            max_context: 262_144,
            ..EngineShape::default()
        };
        let refusal =
            ignis_core::residency::plan_host(&qwen27b_host_request(10 << 30, 1_350_860_800, 297_000_960, 1 << 30, &shape))
                .expect_err("short of the arena");
        let sink = std::sync::Arc::new(ignis_logging::MemorySink::new());
        let subscriber = tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone()));
        tracing::subscriber::with_default(subscriber, || {
            log_qwen27b_host_plan_refused(&refusal, Some(ignis_core::VisionResidency::Host))
        });
        let events: Vec<serde_json::Value> =
            sink.lines().iter().map(|line| serde_json::from_str(line).expect("valid json")).collect();
        assert_eq!(events.len(), 1, "{events:?}");
        let event = &events[0];
        assert_eq!(event["event_name"], "ignis.runtime.qwen27b_host_plan_refused");
        assert_eq!(event["severity_text"], "WARN");
        assert_eq!(event["body"], refusal.to_string());
        let attributes = &event["attributes"];
        for (field, bytes) in [
            ("available_bytes", 10u64 << 30),
            ("embedding_table_bytes", 1_350_860_800),
            ("vision_tower_bytes", 297_000_960),
            ("vision_pool_bytes", 41_943_040),
            ("staging_bytes", 0),
            ("retained_host_slots_bytes", 1 << 30),
            ("kv_ram_arena_bytes", 8 << 30),
            ("margin_bytes", 4 << 30),
        ] {
            assert_eq!(attributes[field], bytes, "{field}: {attributes}");
        }
        assert_eq!(attributes["crossing_line"], "kv_ram_arena");
        assert_eq!(attributes["vision_residency"], "host");
    }

    /// ADR 0075 (GitHub #414): the 27B's host plan event says where the load
    /// keeps its vision tower and how many bytes the tower's own line holds,
    /// and (GitHub #415) the embedding pool's; a text load names no residency.
    #[test]
    fn the_27b_host_plan_event_carries_the_vision_residency_and_the_towers_and_pools_lines() {
        use tracing_subscriber::layer::SubscriberExt;
        let shape = EngineShape {
            host_pool_bytes: 8 << 30,
            retained_host_slots: 4,
            max_context: 262_144,
            ..EngineShape::default()
        };
        let capture = |tower: u64, vision: Option<ignis_core::VisionResidency>| {
            let shape = EngineShape { vision: vision.map(|residency| host_vision(4096).with_residency(residency)), ..shape };
            let request = qwen27b_host_request(64 << 30, 1_350_860_800, tower, 1 << 30, &shape);
            let host = ignis_core::residency::plan_host(&request).expect("64 GiB holds the plan");
            let sink = std::sync::Arc::new(ignis_logging::MemorySink::new());
            let subscriber = tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone()));
            tracing::subscriber::with_default(subscriber, || log_qwen27b_host_plan(&request, &host, &shape, vision));
            let events: Vec<serde_json::Value> =
                sink.lines().iter().map(|line| serde_json::from_str(line).expect("valid json")).collect();
            assert_eq!(events.len(), 1, "{events:?}");
            assert_eq!(events[0]["event_name"], "ignis.runtime.qwen27b_host_plan");
            (events[0]["attributes"].clone(), host.total_bytes)
        };
        let (host, planned) = capture(297_000_960, Some(ignis_core::VisionResidency::Host));
        assert_eq!(host["vision_residency"], "host");
        assert_eq!(host["vision_tower_bytes"], 297_000_960u64);
        assert_eq!(host["vision_pool_bytes"], 41_943_040u64, "one 4,096-token item");
        assert_eq!(host["embedding_table_bytes"], 1_350_860_800u64, "the table's line, not the tower's");
        assert_eq!(host["planned_bytes"], planned);
        assert_eq!(planned, 1_350_860_800 + 297_000_960 + 41_943_040 + (1 << 30) + (8 << 30));
        assert_eq!(host["retained_host_slots"], 4);
        let (device, _) = capture(0, Some(ignis_core::VisionResidency::Device));
        assert_eq!(device["vision_residency"], "device");
        assert_eq!(device["vision_tower_bytes"], 0);
        assert_eq!(device["vision_pool_bytes"], 0, "the device's pool is a VRAM line");
        let (text, _) = capture(0, None);
        assert!(text.get("vision_residency").is_none(), "a text load has no residency: {text}");
        assert_eq!(text["vision_pool_bytes"], 0);
    }

    fn plan_for(mode: ignis_core::VramMode, free: u64) -> ignis_core::VramPlan {
        let reserved = ignis_runtime::ReservedBytes {
            workspace: 1 << 30,
            lane_state: 1 << 30,
            retained_slots: u64::from(crate::config::DEFAULT_RETAINED_DEVICE_SLOTS) * 196_880_384,
            ..Default::default()
        };
        let page_bytes = ignis_core::KvFormat::HqE8_2b.page_bytes(ignis_core::KvGeometry::qwen38_27b());
        let arena = move |pages: u32| 4096 + u64::from(pages) * page_bytes;
        ignis_core::plan_vram(&ignis_core::VramRequest {
            mode,
            free_at_start_bytes: free,
            lines: vram_lines(17 << 30, reserved),
            kv_page_bytes: page_bytes,
            max_context_tokens: 262_144,
            retained_slots: EngineShape::default().retained_slots(),
            kv_pool: None,
            residency: ignis_core::Residency::Resident,
            embedding_pool_named: false,
            kv_arena_bytes: &arena,
            can_page: true,
        })
        .expect("fits")
    }

    #[test]
    fn one_vram_plan_event_carries_every_line_the_mode_and_the_headroom() {
        let plan = plan_for(ignis_core::VramMode::Derived { headroom_bytes: 1 << 30 }, 30 << 30);
        let records = captured_vram_plan(&plan);
        assert_eq!(records.len(), 1, "{records:?}");
        let event = &records[0];
        assert_eq!(event["event_name"], "ignis.runtime.vram_plan", "{event}");
        let field = |name: &str| {
            event
                .get(name)
                .or_else(|| event["attributes"].get(name))
                .cloned()
                .unwrap_or_else(|| panic!("no `{name}` in {event}"))
        };
        assert_eq!(field("mode"), "derived");
        assert_eq!(field("free_at_start_bytes"), 30u64 << 30);
        // GitHub #324: of which this process already held so much when NVML read it.
        assert_eq!(field("process_held_bytes"), PROCESS_HELD);
        assert_eq!(field("headroom_bytes"), 1u64 << 30);
        assert_eq!(field("budget_bytes"), plan.budget_bytes);
        for (name, bytes) in plan.lines.entries() {
            assert_eq!(field(&format!("{name}_bytes")), bytes, "{name}");
        }
        assert_eq!(field("kv_pool_bytes"), plan.kv_pool_bytes);
        // ADR 0045 (AC 7): the branch the pool was sized by, and its size.
        assert_eq!(field("kv_pool_policy"), "resident");
        assert_eq!(field("kv_pool_pages"), plan.kv_page_count);
        assert_eq!(field("kv_pool_tokens"), u64::from(plan.kv_page_count) * 64);
        assert_eq!(field("total_bytes"), plan.total_bytes);
        assert_eq!(
            field("allocated_at_load_bytes"),
            plan.total_bytes,
            "what Task Manager shows right after load: every line, the retained slots included"
        );
        assert_eq!(field("oversubscribed"), false);
        // GitHub #281: the retained slots of both kinds, and the host block
        // that is none of the VRAM lines above.
        assert_eq!(field("retained_device_slots"), DEFAULT_RETAINED.device_slots);
        assert_eq!(field("retained_host_slots"), DEFAULT_RETAINED.host_slots);
        assert_eq!(field("retained_host_bytes"), DEFAULT_RETAINED.host_bytes);
        // ADR 0063 §3: a load that adapted nothing says so.
        assert_eq!(field("adapted"), "");
    }

    /// ADR 0075 (GitHub #414): a vision load's plan event says where its
    /// tower lives, and a text load's names no residency.
    #[test]
    fn the_vram_plan_event_carries_the_vision_residency() {
        use tracing_subscriber::layer::SubscriberExt;
        let plan = plan_for(ignis_core::VramMode::Derived { headroom_bytes: 1 << 30 }, 30 << 30);
        let capture = |vision: Option<ignis_core::VisionResidency>| {
            let sink = std::sync::Arc::new(ignis_logging::MemorySink::new());
            let subscriber = tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone()));
            tracing::subscriber::with_default(subscriber, || log_vram_plan(&plan, DEFAULT_RETAINED, PROCESS_HELD, &[], vision));
            let events: Vec<serde_json::Value> =
                sink.lines().iter().map(|line| serde_json::from_str(line).expect("valid json")).collect();
            assert_eq!(events[0]["event_name"], "ignis.runtime.vram_plan");
            events[0]["attributes"].clone()
        };
        assert_eq!(capture(Some(ignis_core::VisionResidency::Host))["vision_residency"], "host");
        assert_eq!(capture(Some(ignis_core::VisionResidency::Device))["vision_residency"], "device");
        let text = capture(None);
        assert!(text.get("vision_residency").is_none(), "a text load has no residency: {text}");
    }

    /// ADR 0063 §3 (AC 4): the plan event names every adaptation the load
    /// took, in the order taken, as one list.
    #[test]
    fn the_vram_plan_event_names_the_adaptations_the_load_took() {
        use crate::adaptation::{AdaptStep, Room};
        let plan = plan_for(ignis_core::VramMode::Derived { headroom_bytes: 1 << 30 }, 30 << 30);
        let steps = [
            AdaptStep::DraftHead { from: ignis_core::ProposalHead::Shortlist, to: ignis_core::ProposalHead::Full },
            AdaptStep::DecodeLanes { from: 2, to: 1 },
            AdaptStep::MaxContext { from: 262_144, to: 196_608, context_step: 65_536, room: Room::OneLane },
        ];
        let records = captured_adapted_vram_plan(&plan, &steps);
        assert_eq!(records.len(), 1, "{records:?}");
        assert_eq!(
            records[0]["attributes"]["adapted"],
            "draft_head shortlist -> full, decode_lanes 2 -> 1, max_context 262144 -> 196608 (adapt_context_step 65536, one lane)",
            "{}",
            records[0]
        );
    }

    #[test]
    fn make_s_named_pool_tokens_use_each_model_s_own_bytes_per_token() {
        // `make config` turns a named --vram-kv-pool-bytes into tokens in sh
        // (`mk/kv-pool-tokens.sh`, ADR 0045): its per-token costs are the
        // ones the plan divides a byte count by.
        use ignis_core::KvFormat;
        let script = include_str!("../../../mk/kv-pool-tokens.sh");
        let flash = ignis_core::compute::ModelConfig::qwen38_flash_next();
        let qwen = ignis_core::KvGeometry::qwen38_27b();
        for (case, bytes) in [
            ("flash-next:bf16", flash.paged_sections(KvFormat::Bf16).bytes_per_token()),
            ("flash-next:*", flash.paged_sections(KvFormat::HqE8_2b).bytes_per_token()),
            ("27b:bf16", KvFormat::Bf16.bytes_per_token(qwen)),
            ("*", KvFormat::HqE8_2b.bytes_per_token(qwen)),
        ] {
            let line = format!("{case}) per_token={bytes} ;;");
            assert!(script.contains(&line), "`{line}` missing from mk/kv-pool-tokens.sh");
        }
    }

    #[test]
    fn a_reload_plans_the_pool_its_cold_start_planned() {
        // GitHub #324: after a teardown NVML reads the free memory with the
        // context, and the 27B's per-process state once a 27B has loaded,
        // still held. The plan charges both again, so a reload that did not
        // add them back planned 8.2% fewer pages in the AC 14 live run.
        use ignis_runtime::{held_before_load, CUDA_CONTEXT_BYTES, QWEN27B_PROCESS_STATE_BYTES};
        const COLD_FREE: u64 = 31_104_512_000;
        let mode = ignis_core::VramMode::Derived { headroom_bytes: 1_258_291_200 };
        let cold = plan_for(mode, COLD_FREE + held_before_load(false, false));
        for (qwen27b_loaded, held) in
            [(false, CUDA_CONTEXT_BYTES), (true, CUDA_CONTEXT_BYTES + QWEN27B_PROCESS_STATE_BYTES)]
        {
            let reload = plan_for(mode, COLD_FREE - held + held_before_load(true, qwen27b_loaded));
            assert_eq!((reload.kv_page_count, reload.budget_bytes), (cold.kv_page_count, cold.budget_bytes));
            let unadded = plan_for(mode, COLD_FREE - held);
            assert!(unadded.kv_page_count < cold.kv_page_count, "the double charge this adds back");
        }
        assert_eq!(held_before_load(false, false), 0, "a first load reads before its context exists");
        assert_eq!(held_before_load(false, true), 0, "nothing is held without a context");
    }

    #[test]
    fn an_oversubscribed_explicit_plan_logs_the_budget_and_a_warning() {
        let mode = ignis_core::VramMode::Explicit {
            budget_bytes: 31 << 30,
            allow_oversubscription: true,
        };
        let plan = plan_for(mode, 30 << 30);
        let records = captured_vram_plan(&plan);
        assert_eq!(records.len(), 2, "{records:?}");
        let event = &records[0];
        let field = |name: &str| event.get(name).or_else(|| event["attributes"].get(name)).cloned();
        assert_eq!(field("mode").expect("mode"), "explicit");
        assert_eq!(field("budget_bytes").expect("budget"), 31u64 << 30);
        assert_eq!(field("oversubscribed").expect("oversubscribed"), true);
        assert!(field("headroom_bytes").is_none(), "{event}");
        assert_eq!(records[1]["event_name"], "ignis.runtime.vram_oversubscribed");
        assert_eq!(records[1]["severity_text"], "WARN");
    }

    struct StubLeaf;

    impl StepLeaf for StubLeaf {
        type Model = ();
        type Sequence = ();
        type Prefix = ();
        type SnapshotBuf = Vec<u8>;
        type Media = ();
        type Checkpoint = ();

        fn load_model(&self) -> Result<Self::Model, i32> {
            Ok(())
        }
        fn release_model(&self, _model: Self::Model) {}
        fn stats(&self, _model: &Self::Model) -> Result<ignis_runtime::RuntimeStats, i32> {
            Ok(ignis_runtime::RuntimeStats::default())
        }
        fn allocate_sequence(
            &self,
            _model: &Self::Model,
            _context_tokens: u32,
        ) -> Result<Self::Sequence, i32> {
            Ok(())
        }
        fn release_sequence(&self, _model: &Self::Model, _sequence: Self::Sequence) {}
        fn allocate_sequence_shared(
            &self,
            _model: &Self::Model,
            _context_tokens: u32,
            _prefix: &Self::Prefix,
        ) -> Result<Self::Sequence, i32> {
            Ok(())
        }
        fn publish_prefix(
            &self,
            _model: &Self::Model,
            _sequence: &mut Self::Sequence,
            _prefix_tokens: u32,
            _retained_slot: u32,
        ) -> Result<Self::Prefix, i32> {
            Ok(())
        }
        fn release_prefix(&self, _model: &Self::Model, _prefix: Self::Prefix) {}
        fn vocab(&self, _model: &Self::Model) -> u32 {
            8
        }
        fn prefill(
            &self,
            _model: &Self::Model,
            _sequence: &mut Self::Sequence,
            _tokens: &[TokenId],
            _start_position: u32,
            _params: DecodeParams,
            _permitted: &[TokenId],
            _redirect: Option<&mut ignis_runtime::DrawRedirect<'_>>,
            _out_logits: Option<&mut [f32]>,
            _attention: Option<&mut ignis_runtime::AttentionRead>,
        ) -> Result<f32, i32> {
            Ok(0.0)
        }
        fn decode(
            &self,
            _model: &Self::Model,
            sequences: &mut [&mut Self::Sequence],
            _lanes: &[ignis_runtime::DecodeLane<'_>],
        ) -> Result<Vec<ignis_runtime::LaneRun>, i32> {
            Ok(vec![ignis_runtime::LaneRun::token(7); sequences.len()])
        }
        fn alloc_snapshot_buf(&self, bytes: u64) -> Result<Self::SnapshotBuf, i32> {
            Ok(vec![0u8; bytes as usize])
        }
        fn snapshot_bytes(&self, _model: &Self::Model, _sequence: &Self::Sequence) -> Result<u64, i32> {
            Ok(0)
        }
        fn snapshot_into(
            &self,
            _model: &Self::Model,
            _sequence: &Self::Sequence,
            _dst: &mut [u8],
        ) -> Result<(), i32> {
            Ok(())
        }
        fn restore_sequence(
            &self,
            _model: &Self::Model,
            _sequence: &mut Self::Sequence,
            _src: &[u8],
        ) -> Result<(), i32> {
            Ok(())
        }
    }

    #[test]
    fn scheduler_construction_uses_the_runtime_adapter() {
        let model = Arc::new(Model::load(Arc::new(StubLeaf)).unwrap());
        let mut scheduler = scheduler(
            SchedulerConfig {
                model: "stub".into(),
                ..SchedulerConfig::default()
            },
            model,
            99,
        );
        scheduler
            .submit(
                RequestInput {
                    decision: None,
                    constrained: None,
                    forced_literal: None,
                    warm_up: false,
                    multimodal: None,
                    opener_tokens: None,
                    user_turn_tokens: None,
                    system_block_tokens: None,
                    reuse_boundaries: Vec::new(),
                    model: "stub".into(),
                    tokens: vec![1],
                    params: DecodeParams {
                        max_tokens: Some(1),
                        ..DecodeParams::default()
                    },
                },
                RequestClass::Interactive,
            )
            .unwrap();

        assert!(
            scheduler
                .advance()
                .iter()
                .any(|event| matches!(event, ignis_core::SchedEvent::Done { tokens: 1, .. }))
        );
    }
}
