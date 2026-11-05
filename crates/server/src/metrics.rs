//! Prometheus metrics (GitHub #89, ADR 0017): the exposition
//! `ignis-server` serves by default (`--server-metrics false` turns it off) — at `GET /metrics` on its own
//! listener, and at `GET /ui/metrics` on the API listener for the Playground.
//!
//! [`Metrics`] is an aggregate projection of facts the model thread already
//! sends to the asynchronous telemetry consumer (`engine.rs`'s
//! `telemetry_task`). That consumer is its only writer; the HTTP task only
//! reads it, and all text encoding happens there, at scrape time. Nothing in
//! the scheduler, the model thread, the runtime or the kernel leaf knows it
//! exists, and with `--server-metrics false` it is never built.
//!
//! Fixed atomics rather than a lock: a scrape can never make the consumer
//! wait, and the consumer can never make a scrape wait. A scrape therefore
//! reads each series on its own, not one consistent cut across all of them —
//! which Prometheus does not assume either.

use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::Router;
use axum::http::header;
use axum::routing::get;
use ignis_core::checkpoint::{RetainedKind, RetainedStateOperation, ReuseSource};
use ignis_core::scheduler::{DiskOp, DiskSource};
use ignis_core::{RetainedSkip, SubmitError};

/// What one load reserved, and the shapes the reservations bound (GitHub
/// #216, ADR 0030 §Observability). Computed once, during the load that
/// already builds the plan, and never read again — so exporting it adds no
/// serving work of any kind.
///
/// A load that has no plan (the placeholder path, which loads no model) has
/// no reservations either, and every series below stays at its zero.
#[derive(Debug, Clone)]
pub struct LoadReservations {
    /// The plan's lines, as the plan itself holds them — so the exposition
    /// names them with the plan's own spellings rather than a second set
    /// kept in step by hand.
    pub lines: ignis_core::VramLines,
    /// The budget the plan was laid out inside, derived or explicit.
    pub budget_bytes: u64,
    /// The KV pool's page count, as the leaf verified it at load.
    pub kv_pool_pages: u32,
    /// One KV page's bytes.
    pub kv_page_bytes: u64,
    /// `--reuse-kv-host-pool-bytes`, pinned whole at start.
    pub kv_ram_arena_bytes: u64,
    /// The retained slots this load hands out — the effective count, which
    /// is 0 with prompt reuse off and neither `--reuse-retained-device` nor
    /// `--reuse-retained-host` named.
    pub retained_slots: u32,
    /// Of those, the host ones (GitHub #281), and their pinned block: host
    /// memory, beside the plan's lines rather than one of them.
    pub retained_host_slots: u32,
    pub retained_host_bytes: u64,
    /// The KV-disk tier's effective budget (spec vram-budget/03), or `None`
    /// on a load without the tier -- which then never renders a disk series,
    /// as a 27B load never renders the expert residency's.
    pub kv_disk_bytes: Option<u64>,
    }

impl LoadReservations {
    /// What no load reserved: every series at its zero, and no disk tier.
    /// What the gauges read while the idle clock has unloaded the model
    /// (spec model-switch/02, [`Metrics::set_unloaded`]).
    pub(crate) fn none() -> Self {
        Self {
            lines: ignis_core::VramLines::default(),
            budget_bytes: 0,
            kv_pool_pages: 0,
            kv_page_bytes: 0,
            kv_ram_arena_bytes: 0,
            retained_slots: 0,
            retained_host_slots: 0,
            retained_host_bytes: 0,
            kv_disk_bytes: None,
            }
    }

    }

/// The exposition's content type: Prometheus text format 0.0.4.
pub const CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

/// `ignis_request_ttft_seconds`' bucket boundaries (ADR 0017), in
/// milliseconds — the telemetry clock's unit.
const TTFT_BOUNDS_MS: [u64; 12] =
    [50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000, 30_000, 60_000, 120_000, 300_000];

/// `ignis_request_duration_seconds`' bucket boundaries (ADR 0017), in
/// milliseconds.
const DURATION_BOUNDS_MS: [u64; 12] =
    [100, 250, 500, 1_000, 2_500, 5_000, 10_000, 30_000, 60_000, 120_000, 300_000, 600_000];

/// Why a submission was rejected: `ignis_requests_rejected_total`'s fixed
/// `reason` set (ADR 0017).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rejection {
    /// The engine could not admit it right now.
    Full,
    /// It named a model the engine does not load.
    UnknownModel,
    /// It can never fit, however empty the engine is.
    Oversized,
}

impl Rejection {
    /// The reason a submit error counts under. A request longer than the
    /// per-sequence context (GitHub #166, after ADR 0017's table) is a
    /// request that can never fit, like one larger than the KV pool.
    pub fn of(err: &SubmitError) -> Self {
        match err {
            SubmitError::Full => Self::Full,
            SubmitError::UnknownModel(_) => Self::UnknownModel,
            SubmitError::Oversized | SubmitError::ContextExceeded { .. } => Self::Oversized,
        }
    }

    const ALL: [Rejection; 3] = [Self::Full, Self::UnknownModel, Self::Oversized];

    fn label(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::UnknownModel => "unknown_model",
            Self::Oversized => "oversized",
        }
    }
}

/// Which of the two counting routes a call reached:
/// `ignis_tokenize_requests_total`'s fixed `route` label (GitHub #285).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizeRoute {
    /// `POST /v1/tokenize`.
    Tokenize,
    /// `POST /v1/detokenize`.
    Detokenize,
}

impl TokenizeRoute {
    const ALL: [TokenizeRoute; 2] = [Self::Tokenize, Self::Detokenize];

    fn label(self) -> &'static str {
        match self {
            Self::Tokenize => "tokenize",
            Self::Detokenize => "detokenize",
        }
    }
}

/// What the loop guard did about a loop it saw (ADR 0052, 0053):
/// `ignis_loop_guard_loops_total`'s fixed `outcome` label (GitHub #330).
/// One per `ignis.loop_guard.*` log line, named after it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopOutcome {
    /// Taken back and nudged: `rewound` with `closed=false`.
    Rewound,
    /// Taken back and closed, its nudges spent: `rewound` with `closed=true`.
    Closed,
    /// Only logged: `log` mode, or a request with images (`detected`).
    Detected,
    /// Held: a rewind would keep too little reasoning (`too_little_kept`).
    TooLittleKept,
    /// Not taken back: `max_tokens` leaves no continuation room (`no_room`).
    NoRoom,
    /// Not taken back: the nudge cannot be encoded (`no_tokenizer`).
    NoTokenizer,
    /// Cancelled, and its continuation refused (`submit_refused`).
    SubmitRefused,
}

impl LoopOutcome {
    const ALL: [LoopOutcome; 7] =
        [Self::Rewound, Self::Closed, Self::Detected, Self::TooLittleKept, Self::NoRoom, Self::NoTokenizer, Self::SubmitRefused];

    fn label(self) -> &'static str {
        match self {
            Self::Rewound => "rewound",
            Self::Closed => "closed",
            Self::Detected => "detected",
            Self::TooLittleKept => "too_little_kept",
            Self::NoRoom => "no_room",
            Self::NoTokenizer => "no_tokenizer",
            Self::SubmitRefused => "submit_refused",
        }
    }
}

/// `ignis_loop_guard_min_ratio`'s bucket boundaries (GitHub #330), in
/// millionths: a request's lowest loop score. Where the signal is: the
/// guard's threshold (0.265 by default) and the measured margin around it,
/// harmful loops at most 0.257 and agentic reasoning at least 0.274
/// (`docs/findings/2026-10-16-a-loop-persists.md`); plain prose scores
/// 0.45-0.49.
const LOOP_GUARD_MIN_RATIO_BOUNDS: [u64; 10] =
    [100_000, 150_000, 200_000, 250_000, 300_000, 350_000, 400_000, 450_000, 500_000, 1_000_000];


/// A fixed-bucket histogram over observations in `[0, 1]`, counted in
/// millionths.
///
/// Millionths rather than `f64` for the same reason the latency histograms
/// count milliseconds: a sum of integers is exact and a sum of floats is
/// whatever the order of arrival made it, and two scrapes of the same
/// observations must agree.
#[derive(Debug)]
struct Ratio {
    bounds: &'static [u64; 10],
    /// Observations per bucket, not cumulative; the last is `+Inf`'s own.
    buckets: [AtomicU64; 11],
    sum_millionths: AtomicU64,
}

impl Ratio {
    fn new(bounds: &'static [u64; 10]) -> Self {
        Self { bounds, buckets: Default::default(), sum_millionths: AtomicU64::new(0) }
    }

    /// Observe `value`, a probability.
    ///
    /// Anything that is **not** a probability — negative, above one, or not
    /// a number — is counted in `+Inf` and nowhere else. It is not clamped:
    /// the last bound is `1`, so `le="1"` equalling `_count` is the
    /// exposition's own statement that every reading was in range, and
    /// clamping would make that statement true by construction and
    /// therefore worth nothing. `_count - le="1"` is the number of
    /// readings that were not probabilities.
    ///
    /// It is not dropped either, because a readout producing a NaN is a
    /// failure of exactly the kind this histogram exists to show, and a
    /// clamp to zero would file it under "the options collapsed" — a
    /// different diagnosis with a different cause.
    ///
    /// `_sum` takes the value when it is one a sum can hold, so a mass of
    /// 1.5 inflates the mean the way it should; a NaN or a negative
    /// contributes nothing to it, and the bucket count is where they are
    /// visible.
    fn observe(&self, value: f64) {
        let millionths = match value.is_finite() && value >= 0.0 {
            true => (value * 1e6).round() as u64,
            false => 0,
        };
        let in_range = value.is_finite() && (0.0..=1.0).contains(&value);
        let bucket = match in_range {
            true => self
                .bounds
                .iter()
                .position(|&bound| millionths <= bound)
                .unwrap_or(self.bounds.len()),
            false => self.bounds.len(),
        };
        self.buckets[bucket].fetch_add(1, Ordering::Relaxed);
        self.sum_millionths.fetch_add(millionths, Ordering::Relaxed);
    }

    fn render(&self, out: &mut String, name: &str, help: &str) {
        render_buckets(out, name, help, self.bounds, &self.buckets, &self.sum_millionths, MILLIONTHS);
    }
}

/// The scale an observation is counted in: thousandths of a second for the
/// latency histograms, millionths for a ratio.
const THOUSANDTHS: u64 = 1_000;
const MILLIONTHS: u64 = 1_000_000;

/// Cumulative `_bucket` lines, then `_sum` and `_count`, for a histogram
/// whose observations are integers of `1 / scale`.
///
/// One function for both shapes because the exposition format is the
/// format whatever the unit is — the two histogram types differ in their
/// bucket count, their scale and what they accept, and not in a single
/// character of what they print. The count is the `+Inf` bucket as read
/// here, so `_count` and the last bucket always agree.
fn render_buckets(
    out: &mut String,
    name: &str,
    help: &str,
    bounds: &[u64],
    buckets: &[AtomicU64],
    sum: &AtomicU64,
    scale: u64,
) {
    declare(out, name, "histogram", help);
    let mut cumulative = 0;
    for (bucket, count) in buckets.iter().enumerate() {
        cumulative += count.load(Ordering::Relaxed);
        let le = match bounds.get(bucket) {
            Some(&bound) => decimal(bound, scale),
            None => "+Inf".to_owned(),
        };
        let _ = writeln!(out, "{name}_bucket{{le=\"{le}\"}} {cumulative}");
    }
    let _ = writeln!(out, "{name}_sum {}", decimal(sum.load(Ordering::Relaxed), scale));
    let _ = writeln!(out, "{name}_count {cumulative}");
}

/// `value` units of `1 / scale`, as a decimal without trailing zeros.
fn decimal(value: u64, scale: u64) -> String {
    let (whole, frac) = (value / scale, value % scale);
    if frac == 0 {
        return whole.to_string();
    }
    let digits = scale.ilog10() as usize;
    let frac = format!("{frac:0digits$}");
    format!("{whole}.{}", frac.trim_end_matches('0'))
}

/// A fixed-bucket histogram over millisecond observations.
#[derive(Debug)]
struct Histogram {
    bounds_ms: &'static [u64; 12],
    /// Observations per bucket, not cumulative; the last is `+Inf`'s own.
    buckets: [AtomicU64; 13],
    sum_ms: AtomicU64,
}

impl Histogram {
    fn new(bounds_ms: &'static [u64; 12]) -> Self {
        Self { bounds_ms, buckets: Default::default(), sum_ms: AtomicU64::new(0) }
    }

    fn observe(&self, ms: u64) {
        let bucket = self.bounds_ms.iter().position(|&bound| ms <= bound).unwrap_or(self.bounds_ms.len());
        self.buckets[bucket].fetch_add(1, Ordering::Relaxed);
        self.sum_ms.fetch_add(ms, Ordering::Relaxed);
    }

    fn render(&self, out: &mut String, name: &str, help: &str) {
        render_buckets(out, name, help, self.bounds_ms, &self.buckets, &self.sum_ms, THOUSANDTHS);
    }
}

/// The aggregate the telemetry consumer maintains while metrics are on —
/// except the rejections, which the HTTP handler records after the submit
/// call returns its error (ADR 0017).
#[derive(Debug)]
pub struct Metrics {
    waiting: AtomicU64,
    running: AtomicU64,
    accepted: AtomicU64,
    completed: AtomicU64,
    cancelled: AtomicU64,
    rejected: [AtomicU64; 3],
    generated_tokens: AtomicU64,
    decoded_tokens: AtomicU64,
    kv_evictions: AtomicU64,
    kv_ram_evictions: AtomicU64,
    prefix_reused_tokens: AtomicU64,
    /// Per [`ReuseSource::index`], for each of the families below (#190).
    retained_reused_tokens: RetainedFamily,
    retained_state_hits: RetainedFamily,
    retained_state_misses: RetainedFamily,
    retained_state_spills: RetainedFamily,
    retained_state_discards: RetainedFamily,
    retained_state_restores: RetainedFamily,
    /// Per [`RetainedSkip::index`] (GitHub #216).
    retained_slot_skips: [AtomicU64; RetainedSkip::ALL.len()],
    /// The retained slots held right now, and the count this load hands out.
    retained_slots_in_use: AtomicU64,
    retained_slots_capacity: AtomicU64,
    /// Of the retained slots, the host ones and their pinned block (GitHub
    /// #281): written once at load.
    retained_host_slots: AtomicU64,
    retained_host_bytes: AtomicU64,
    /// The load's reservations (GitHub #216): the plan's lines in
    /// `VramLines::entries()` order, then the shapes they bound. Written once
    /// at load, and zero on a load that built no plan.
    vram_reserved: [AtomicU64; VRAM_LINE_COUNT],
    vram_budget_bytes: AtomicU64,
    kv_pool_pages: AtomicU64,
    kv_page_bytes: AtomicU64,
    kv_ram_arena_capacity_bytes: AtomicU64,
    /// What the scheduler has occupied, republished on every tick.
    kv_pool_used_pages: AtomicU64,
    kv_ram_arena_used_bytes: AtomicU64,
    /// KV-disk (spec vram-budget/03): 1 while the load has the tier, its
    /// budget and what it holds, its live spills per [`DiskSource`] and its
    /// failures per [`DiskOp`] (in their declaration order).
    kv_disk_loaded: AtomicU64,
    kv_disk_capacity_bytes: AtomicU64,
    kv_disk_used_bytes: AtomicU64,
    kv_disk_spills: [AtomicU64; 2],
    kv_disk_failures: [AtomicU64; 2],
    ttft: Histogram,
    duration: Histogram,
    /// Requests whose reasoning block the thinking budget closed (spec
    /// server/08). Absent from the exposition until the first one, like the
    /// decision family.
    thinking_forced_closes: AtomicU64,
    /// The loop guard (GitHub #330): the requests it watched and those that
    /// looped, its loops per [`LoopOutcome::ALL`], the tokens its rewinds
    /// took back, and each watched request's lowest score. Absent from the
    /// exposition until a watched request ended or a loop was seen, then
    /// whole.
    loop_guard_watched: AtomicU64,
    loop_guard_looped: AtomicU64,
    loop_guard_loops: [AtomicU64; LoopOutcome::ALL.len()],
    loop_guard_rewound_tokens: AtomicU64,
    loop_guard_min_ratio: Ratio,
    /// Speculative decoding (P5-06's request counters, GitHub #307): verify
    /// rounds, drafts verified and drafts committed, and per draft position
    /// how many rounds proposed it and committed it.
    spec_rounds: AtomicU64,
    spec_drafted: AtomicU64,
    spec_accepted: AtomicU64,
    spec_drafted_at: [AtomicU64; ignis_core::types::DRAFT_POSITIONS],
    spec_accepted_at: [AtomicU64; ignis_core::types::DRAFT_POSITIONS],
    /// Per [`TokenizeRoute::ALL`] (GitHub #285). Absent from the exposition
    /// until one of them moves. Counts calls, not requests of the lifecycle
    /// above: a counting route never enters the scheduler.
    tokenize_calls: [AtomicU64; TokenizeRoute::ALL.len()],
    /// Open Responses API WebSockets, and their requests waiting in the
    /// server-wide admission queue (GitHub #282).
    responses_sockets: AtomicU64,
    responses_queued: AtomicU64,
    }

/// One retained-state family: a count per residency tier and per kind of
/// retained state (GitHub #216) — `[tier][kind]`, so a prompt checkpoint and
/// a shared prefix are never summed into one figure.
type RetainedFamily = [[AtomicU64; RetainedKind::ALL.len()]; ReuseSource::ALL.len()];

/// How many lines a VRAM plan has, taken from the plan's own shape rather
/// than written out again here.
const VRAM_LINE_COUNT: usize = ignis_core::VramLines::LINES;

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    /// An all-zero projection.
    pub fn new() -> Self {
        Self {
            waiting: AtomicU64::new(0),
            running: AtomicU64::new(0),
            accepted: AtomicU64::new(0),
            completed: AtomicU64::new(0),
            cancelled: AtomicU64::new(0),
            rejected: Default::default(),
            generated_tokens: AtomicU64::new(0),
            decoded_tokens: AtomicU64::new(0),
            kv_evictions: AtomicU64::new(0),
            kv_ram_evictions: AtomicU64::new(0),
            prefix_reused_tokens: AtomicU64::new(0),
            retained_reused_tokens: Default::default(),
            retained_state_hits: Default::default(),
            retained_state_misses: Default::default(),
            retained_state_spills: Default::default(),
            retained_state_discards: Default::default(),
            retained_state_restores: Default::default(),
            retained_slot_skips: Default::default(),
            retained_slots_in_use: AtomicU64::new(0),
            retained_slots_capacity: AtomicU64::new(0),
            retained_host_slots: AtomicU64::new(0),
            retained_host_bytes: AtomicU64::new(0),
            vram_reserved: Default::default(),
            vram_budget_bytes: AtomicU64::new(0),
            kv_pool_pages: AtomicU64::new(0),
            kv_page_bytes: AtomicU64::new(0),
            kv_ram_arena_capacity_bytes: AtomicU64::new(0),
            kv_pool_used_pages: AtomicU64::new(0),
            kv_ram_arena_used_bytes: AtomicU64::new(0),
            kv_disk_loaded: AtomicU64::new(0),
            kv_disk_capacity_bytes: AtomicU64::new(0),
            kv_disk_used_bytes: AtomicU64::new(0),
            kv_disk_spills: Default::default(),
            kv_disk_failures: Default::default(),
            ttft: Histogram::new(&TTFT_BOUNDS_MS),
            duration: Histogram::new(&DURATION_BOUNDS_MS),
            thinking_forced_closes: AtomicU64::new(0),
            loop_guard_watched: AtomicU64::new(0),
            loop_guard_looped: AtomicU64::new(0),
            loop_guard_loops: Default::default(),
            loop_guard_rewound_tokens: AtomicU64::new(0),
            loop_guard_min_ratio: Ratio::new(&LOOP_GUARD_MIN_RATIO_BOUNDS),
            spec_rounds: AtomicU64::new(0),
            spec_drafted: AtomicU64::new(0),
            spec_accepted: AtomicU64::new(0),
            spec_drafted_at: std::array::from_fn(|_| AtomicU64::new(0)),
            spec_accepted_at: std::array::from_fn(|_| AtomicU64::new(0)),
            tokenize_calls: Default::default(),
            responses_sockets: AtomicU64::new(0),
            responses_queued: AtomicU64::new(0),
            }
    }

    /// A request that ran verify rounds ended: its speculative counters
    /// (GitHub #307), the request log's `spec.*` summed across requests.
    pub(crate) fn record_speculation(&self, spec: &ignis_core::types::SpecCounters) {
        self.spec_rounds.fetch_add(u64::from(spec.rounds), Ordering::Relaxed);
        self.spec_drafted.fetch_add(u64::from(spec.drafted), Ordering::Relaxed);
        self.spec_accepted.fetch_add(u64::from(spec.accepted), Ordering::Relaxed);
        for (series, &n) in self.spec_drafted_at.iter().zip(&spec.drafted_at) {
            series.fetch_add(u64::from(n), Ordering::Relaxed);
        }
        for (series, &n) in self.spec_accepted_at.iter().zip(&spec.accepted_at) {
            series.fetch_add(u64::from(n), Ordering::Relaxed);
        }
    }

    /// A completed request's thinking budget forced the model's close (spec
    /// server/08).
    pub(crate) fn record_thinking_forced_close(&self) {
        self.thinking_forced_closes.fetch_add(1, Ordering::Relaxed);
    }

    /// A request the loop guard watched ended (GitHub #330). Recorded by the
    /// guard, off the model thread, like everything below.
    pub(crate) fn record_loop_guard_watched(&self) {
        self.loop_guard_watched.fetch_add(1, Ordering::Relaxed);
    }

    /// That request had looped.
    pub(crate) fn record_loop_guard_looped(&self) {
        self.loop_guard_looped.fetch_add(1, Ordering::Relaxed);
    }

    /// The guard saw a loop and did `outcome` about it.
    pub(crate) fn record_loop_guard_loop(&self, outcome: LoopOutcome) {
        // `ALL` lists the outcomes in declaration order.
        self.loop_guard_loops[outcome as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// A rewind took `tokens` generated tokens back.
    pub(crate) fn record_loop_guard_rewound_tokens(&self, tokens: u32) {
        self.loop_guard_rewound_tokens.fetch_add(u64::from(tokens), Ordering::Relaxed);
    }

    /// A watched request ended, the lowest of its scores `ratio`.
    pub(crate) fn observe_loop_guard_min_ratio(&self, ratio: f64) {
        self.loop_guard_min_ratio.observe(ratio);
    }

    /// A call reached `/v1/tokenize` or `/v1/detokenize` (GitHub #285),
    /// counted whatever it answered: the series says the route is used, and
    /// a client polling it is what an operator looks for. Recorded by the
    /// handler, like [`Metrics::record_rejected`], since the call has no fact
    /// on the model thread's stream — it never reaches it.
    pub(crate) fn record_tokenize(&self, route: TokenizeRoute) {
        // `ALL` lists the routes in declaration order.
        self.tokenize_calls[route as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// A submission was rejected, for `reason`.
    pub fn record_rejected(&self, reason: Rejection) {
        // `ALL` lists the reasons in declaration order, so a reason's
        // discriminant is its slot.
        self.rejected[reason as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// A request was evicted to the host KV-RAM tier.
    pub(crate) fn record_eviction(&self) {
        self.kv_evictions.fetch_add(1, Ordering::Relaxed);
    }

    /// A live snapshot was dropped out of the KV-RAM tier to make room for
    /// another (GitHub #224) — the tier's own eviction, the mirror of
    /// [`Metrics::record_eviction`]'s departure from the device.
    pub(crate) fn record_kv_ram_eviction(&self) {
        self.kv_ram_evictions.fetch_add(1, Ordering::Relaxed);
    }

    /// A live snapshot was written to KV-disk from `from` (spec
    /// vram-budget/03): straight from the device, or demoted from KV-RAM.
    pub(crate) fn record_kv_disk_spill(&self, from: DiskSource) {
        let at = match from {
            DiskSource::Device => 0,
            DiskSource::KvRam => 1,
        };
        self.kv_disk_spills[at].fetch_add(1, Ordering::Relaxed);
    }

    /// A KV-disk transfer failed (spec vram-budget/03).
    pub(crate) fn record_kv_disk_failure(&self, op: DiskOp) {
        let at = match op {
            DiskOp::Write => 0,
            DiskOp::Read => 1,
        };
        self.kv_disk_failures[at].fetch_add(1, Ordering::Relaxed);
    }

    /// A request's prefill skipped `tokens` through a sibling's prefix.
    pub(crate) fn record_prefix_reused(&self, tokens: u32) {
        self.prefix_reused_tokens.fetch_add(u64::from(tokens), Ordering::Relaxed);
    }

    /// A request's prefill skipped `tokens` through retained state in
    /// `source` — a retained prefix, or a prompt checkpoint (GitHub #190,
    /// #216).
    pub(crate) fn record_retained_reused(&self, source: ReuseSource, kind: RetainedKind, tokens: u32) {
        self.retained_reused_tokens[source.index()][kind.index()]
            .fetch_add(u64::from(tokens), Ordering::Relaxed);
    }

    /// One retained-state lifecycle operation, in its residency tier and on
    /// the kind of state it moved (GitHub #190, #216). The telemetry consumer
    /// is the only writer.
    pub(crate) fn record_retained_state(
        &self,
        operation: RetainedStateOperation,
        source: ReuseSource,
        kind: RetainedKind,
    ) {
        let series = match operation {
            RetainedStateOperation::Hit => &self.retained_state_hits,
            RetainedStateOperation::Miss => &self.retained_state_misses,
            RetainedStateOperation::Spill => &self.retained_state_spills,
            RetainedStateOperation::Discard => &self.retained_state_discards,
            RetainedStateOperation::Restore => &self.retained_state_restores,
        };
        series[source.index()][kind.index()].fetch_add(1, Ordering::Relaxed);
    }

    /// A publish or a capture found no retained slot, or no tail page
    /// (GitHub #216): the signal that a load has run out of room to leave
    /// reuse behind and is running on without it.
    pub(crate) fn record_retained_slot_skip(&self, skip: RetainedSkip) {
        self.retained_slot_skips[skip.index()].fetch_add(1, Ordering::Relaxed);
    }

    /// The retained slots a prefix or checkpoint image holds right now.
    pub(crate) fn set_retained_slots_in_use(&self, in_use: u32) {
        self.retained_slots_in_use.store(u64::from(in_use), Ordering::Relaxed);
    }

    /// What the scheduler had occupied when the last step ended (GitHub
    /// #216): the main pool's pages, and the KV-RAM arena's bytes. Both are
    /// republished on every tick, including the tick of the step that
    /// released the last request — which is the last tick there is, so a
    /// projection that skipped it would read a stale figure for as long as
    /// the load stayed idle.
    pub(crate) fn set_occupancy(&self, occupancy: ignis_core::Occupancy) {
        self.kv_pool_used_pages.store(u64::from(occupancy.kv_used_pages), Ordering::Relaxed);
        self.kv_ram_arena_used_bytes.store(occupancy.kv_ram_used_bytes, Ordering::Relaxed);
        self.kv_disk_used_bytes.store(occupancy.kv_disk_used_bytes, Ordering::Relaxed);
    }

    /// What this load reserved (GitHub #216, ADR 0030): written once, by the
    /// load that built the plan, before the first request is served.
    pub(crate) fn set_load_reservations(&self, reserved: LoadReservations) {
        for (slot, (_, bytes)) in self.vram_reserved.iter().zip(reserved.lines.entries()) {
            slot.store(bytes, Ordering::Relaxed);
        }
        self.vram_budget_bytes.store(reserved.budget_bytes, Ordering::Relaxed);
        self.kv_pool_pages.store(u64::from(reserved.kv_pool_pages), Ordering::Relaxed);
        self.kv_page_bytes.store(reserved.kv_page_bytes, Ordering::Relaxed);
        self.kv_ram_arena_capacity_bytes.store(reserved.kv_ram_arena_bytes, Ordering::Relaxed);
        self.retained_slots_capacity.store(u64::from(reserved.retained_slots), Ordering::Relaxed);
        self.retained_host_slots.store(u64::from(reserved.retained_host_slots), Ordering::Relaxed);
        self.retained_host_bytes.store(reserved.retained_host_bytes, Ordering::Relaxed);
        self.kv_disk_loaded.store(u64::from(reserved.kv_disk_bytes.is_some()), Ordering::Relaxed);
        self.kv_disk_capacity_bytes.store(reserved.kv_disk_bytes.unwrap_or(0), Ordering::Relaxed);
    }

    /// Nothing is loaded (spec model-switch/02: the idle clock unloaded the
    /// model, and the card is free): what the last load reserved, what was
    /// occupied of it and its own counters read as no load at all, until the
    /// next load's reservations replace them. Written after the teardown has
    /// joined the model thread of an engine that sat idle for
    /// `model.idle_unload` seconds, so no tick of its is left to write over
    /// it.
    pub(crate) fn set_unloaded(&self) {
        self.set_load_reservations(LoadReservations::none());
        self.set_occupancy(ignis_core::Occupancy::default());
        self.set_retained_slots_in_use(0);
        self.set_scheduler_requests(0, 0);
        }

    /// A request's first token came `ms` after its submission.
    pub(crate) fn observe_ttft_ms(&self, ms: u64) {
        self.ttft.observe(ms);
    }

    /// A request completed `ms` after its submission.
    pub(crate) fn observe_duration_ms(&self, ms: u64) {
        self.duration.observe(ms);
    }

    /// A submission was accepted by the scheduler.
    pub(crate) fn record_accepted(&self) {
        self.accepted.fetch_add(1, Ordering::Relaxed);
    }

    /// A request completed, having generated `tokens`.
    pub(crate) fn record_completed(&self, tokens: u32) {
        self.completed.fetch_add(1, Ordering::Relaxed);
        self.generated_tokens.fetch_add(u64::from(tokens), Ordering::Relaxed);
    }

    /// A request in flight was dealt one generated token (GitHub #165): the
    /// live counterpart of the tokens `record_completed` adds only at the end.
    pub(crate) fn record_decoded_token(&self) {
        self.decoded_tokens.fetch_add(1, Ordering::Relaxed);
    }

    /// An accepted request was cancelled before it completed (its client
    /// went away).
    pub(crate) fn record_cancelled(&self) {
        self.cancelled.fetch_add(1, Ordering::Relaxed);
    }

    /// The Responses WebSockets open now (GitHub #282).
    pub(crate) fn set_responses_sockets(&self, open: u64) {
        self.responses_sockets.store(open, Ordering::Relaxed);
    }

    /// The socket requests waiting in the admission queue now (GitHub #282).
    pub(crate) fn set_responses_queued(&self, queued: u64) {
        self.responses_queued.store(queued, Ordering::Relaxed);
    }

    /// The scheduler's current request counts by observable state.
    pub(crate) fn set_scheduler_requests(&self, waiting: u32, running: u32) {
        self.waiting.store(u64::from(waiting), Ordering::Relaxed);
        self.running.store(u64::from(running), Ordering::Relaxed);
    }

    /// The Prometheus text exposition of the latest projection.
    pub fn render(&self) -> String {
        let read = |series: &AtomicU64| series.load(Ordering::Relaxed);
        let mut out = String::with_capacity(1024);
        declare(&mut out, "ignis_build_info", "gauge", "Constant build identity with value 1.");
        let _ = writeln!(
            out,
            "ignis_build_info{{version=\"{}\"}} 1",
            escape_label_value(env!("CARGO_PKG_VERSION"))
        );
        declare(
            &mut out,
            "ignis_scheduler_requests",
            "gauge",
            "Current requests by observable scheduler state.",
        );
        let _ = writeln!(out, "ignis_scheduler_requests{{state=\"waiting\"}} {}", read(&self.waiting));
        let _ = writeln!(out, "ignis_scheduler_requests{{state=\"running\"}} {}", read(&self.running));
        let counters = [
            ("ignis_requests_accepted_total", "Accepted submissions.", &self.accepted),
            ("ignis_requests_completed_total", "Completed requests.", &self.completed),
            (
                "ignis_requests_cancelled_total",
                "Accepted requests cancelled before completion.",
                &self.cancelled,
            ),
            (
                "ignis_generated_tokens_total",
                "Generated tokens on completed requests.",
                &self.generated_tokens,
            ),
            (
                "ignis_decoded_tokens_total",
                "Tokens generated so far, counted as each one is emitted.",
                &self.decoded_tokens,
            ),
            ("ignis_kv_cache_evictions_total", "Cumulative host-tier evictions.", &self.kv_evictions),
            // The other end of the same tier (#224): what KV-RAM itself
            // gave up. A live snapshot dropped here is the costliest
            // departure in the system — the request re-prefills from zero —
            // so it is its own series and never folded into the row above,
            // which counts arrivals at the tier.
            (
                "ignis_kv_ram_evictions_total",
                "Live host-tier snapshots dropped from KV-RAM to make room; the request re-prefills from the start.",
                &self.kv_ram_evictions,
            ),
            // Sibling-prefix reuse only, as ADR 0017's table row says. A
            // retained prefix is claimed through the same path (#188), and
            // `SchedEvent::PrefixReused::retained` is what keeps its tokens
            // out of here and in `ignis_retained_reused_tokens_total` (#190).
            (
                "ignis_prefix_reused_tokens_total",
                "Cumulative tokens skipped through sibling-prefix reuse.",
                &self.prefix_reused_tokens,
            ),
        ];
        for (name, help, series) in counters {
            declare(&mut out, name, "counter", help);
            let _ = writeln!(out, "{name} {}", read(series));
        }
        for (name, help, series) in [
            (
                "ignis_retained_reused_tokens_total",
                "Cumulative tokens skipped through retained state, by residency tier.",
                &self.retained_reused_tokens,
            ),
            (
                "ignis_retained_state_hits_total",
                "Retained state a request resumed from, by residency tier.",
                &self.retained_state_hits,
            ),
            // Each `kind` is its own lookup's (GitHub #216, #222): the
            // checkpoint pool's, and the prefix walk's beside it, which
            // misses a tier no *retained* prefix matched in (a live sibling's
            // head is no hit, so it stops no miss) and looks only on the
            // device and in KV-RAM, so `tier="disk",kind="prefix"` stays at
            // zero. ADR 0017 records the rule under its metric table.
            (
                "ignis_retained_state_misses_total",
                "Prefill runs whose first chunk landed with no retained checkpoint, or no retained prefix, matching in the tier (a re-queued request counts again).",
                &self.retained_state_misses,
            ),
            (
                "ignis_retained_state_spills_total",
                "Retained checkpoints and prefixes spilled into the tier.",
                &self.retained_state_spills,
            ),
            (
                "ignis_retained_state_discards_total",
                "Retained checkpoints and prefixes discarded from the tier.",
                &self.retained_state_discards,
            ),
            (
                "ignis_retained_state_restores_total",
                "Retained state restored from the tier.",
                &self.retained_state_restores,
            ),
        ] {
            declare(&mut out, name, "counter", help);
            // Spec vram-budget/03: the disk's rows only on a load that has
            // the tier, as its own families below.
            let disk = read(&self.kv_disk_loaded) != 0;
            for (source, tier) in ReuseSource::ALL.iter().zip(series) {
                if *source == ReuseSource::Disk && !disk {
                    continue;
                }
                for (kind, slot) in RetainedKind::ALL.iter().zip(tier) {
                    let _ = writeln!(
                        out,
                        "{name}{{tier=\"{}\",kind=\"{}\"}} {}",
                        source.as_str(),
                        kind.as_str(),
                        read(slot)
                    );
                }
            }
        }
        declare(
            &mut out,
            "ignis_retained_slot_skips_total",
            "counter",
            "Publishes and captures that found no retained slot or no tail page.",
        );
        for (skip, series) in RetainedSkip::ALL.iter().zip(&self.retained_slot_skips) {
            let _ = writeln!(
                out,
                "ignis_retained_slot_skips_total{{reason=\"{}\"}} {}",
                skip.as_str(),
                read(series)
            );
        }
        // What the load reserved, and what is occupied of it (GitHub #216,
        // ADR 0030 §Observability). Bytes, pages and slots — never a
        // percentage, which would hide which of its two terms moved.
        declare(
            &mut out,
            "ignis_vram_reserved_bytes",
            "gauge",
            "Device bytes this load reserved, by the plan line that reserved them.",
        );
        let names = ignis_core::VramLines::default().entries();
        for ((line, _), series) in names.iter().zip(&self.vram_reserved) {
            let _ = writeln!(out, "ignis_vram_reserved_bytes{{line=\"{line}\"}} {}", read(series));
        }
        let plain_gauges = [
            (
                "ignis_vram_budget_bytes",
                "The device budget the plan was laid out inside.",
                &self.vram_budget_bytes,
            ),
            ("ignis_kv_pool_pages", "Pages the KV pool holds.", &self.kv_pool_pages),
            ("ignis_kv_page_bytes", "One KV page's bytes.", &self.kv_page_bytes),
            (
                "ignis_kv_pool_used_pages",
                "KV pool pages reserved by running requests and retained state.",
                &self.kv_pool_used_pages,
            ),
            (
                "ignis_retained_host_slots",
                "Of the retained slots, those whose images live in the pinned host block.",
                &self.retained_host_slots,
            ),
            (
                "ignis_retained_host_bytes",
                "The pinned host block holding the host retained slots' images.",
                &self.retained_host_bytes,
            ),
            (
                "ignis_responses_sockets",
                "Open Responses API WebSocket connections.",
                &self.responses_sockets,
            ),
            (
                "ignis_responses_queued_requests",
                "Responses WebSocket requests waiting in the server-wide admission queue.",
                &self.responses_queued,
            ),
        ];
        for (name, help, series) in plain_gauges {
            declare(&mut out, name, "gauge", help);
            let _ = writeln!(out, "{name} {}", read(series));
        }
        // Two states of one family: what bounds it, and what is in it. The
        // arena is bytes and says `used`; the slots are counted and say
        // `in_use` (ADR 0030 §Observability spells each).
        for (name, help, used_state, capacity, used) in [
            (
                "ignis_kv_ram_arena_bytes",
                "The pinned host KV-RAM arena: what it holds, and what is used of it.",
                "used",
                &self.kv_ram_arena_capacity_bytes,
                &self.kv_ram_arena_used_bytes,
            ),
            (
                "ignis_retained_slots",
                "Retained slots this load hands out, and how many hold an image.",
                "in_use",
                &self.retained_slots_capacity,
                &self.retained_slots_in_use,
            ),
        ] {
            declare(&mut out, name, "gauge", help);
            let _ = writeln!(out, "{name}{{state=\"capacity\"}} {}", read(capacity));
            let _ = writeln!(out, "{name}{{state=\"{used_state}\"}} {}", read(used));
        }
        // KV-disk (spec vram-budget/03, ADR 0045): every series from the
        // first scrape, zeros included, on a load with the tier -- and none
        // on a load without it.
        if read(&self.kv_disk_loaded) != 0 {
            declare(
                &mut out,
                "ignis_kv_disk_bytes",
                "gauge",
                "The KV-disk tier: its effective budget, and the bytes its files hold.",
            );
            let _ = writeln!(out, "ignis_kv_disk_bytes{{state=\"capacity\"}} {}", read(&self.kv_disk_capacity_bytes));
            let _ = writeln!(out, "ignis_kv_disk_bytes{{state=\"used\"}} {}", read(&self.kv_disk_used_bytes));
            declare(
                &mut out,
                "ignis_kv_disk_spills_total",
                "counter",
                "Live snapshots written to KV-disk, by the tier they came from.",
            );
            for (from, series) in [DiskSource::Device, DiskSource::KvRam].iter().zip(&self.kv_disk_spills) {
                let _ = writeln!(out, "ignis_kv_disk_spills_total{{from=\"{}\"}} {}", from.as_str(), read(series));
            }
            declare(
                &mut out,
                "ignis_kv_disk_failures_total",
                "counter",
                "KV-disk writes refused or failed (nothing lost), and files that failed their check on the way back.",
            );
            for (op, series) in [DiskOp::Write, DiskOp::Read].iter().zip(&self.kv_disk_failures) {
                let _ = writeln!(out, "ignis_kv_disk_failures_total{{op=\"{}\"}} {}", op.as_str(), read(series));
            }
        }
        declare(
            &mut out,
            "ignis_requests_rejected_total",
            "counter",
            "Rejected submissions by fixed reason.",
        );
        for (reason, series) in Rejection::ALL.iter().zip(&self.rejected) {
            let _ = writeln!(out, "ignis_requests_rejected_total{{reason=\"{}\"}} {}", reason.label(), read(series));
        }
        self.ttft.render(&mut out, "ignis_request_ttft_seconds", "Submission-to-first-token latency.");
        self.duration.render(&mut out, "ignis_request_duration_seconds", "Submission-to-completion latency.");
        // The decision family, **only once there has been a decision**
        // (GitHub #241). ADR 0017 exports zeros for everything else; this
        // follows its other rule instead — "only authoritative values are
        // exported" — because a load that serves no decisions is the normal
        // one, and three permanently-zero series plus an eleven-bucket
        // histogram on every scrape of every server would be clutter that
        // says nothing. A `rate()` over a series that appears mid-window is
        // handled by Prometheus the same way a new target is.
        // GitHub #278: absent until the first `locate`, for the decision
        // family's reason.
        // Spec server/08: absent until the thinking budget has forced a
        // close, for the decision family's reason — a load whose traffic
        // never reaches its budget is a normal one, and says nothing by
        // exporting a zero.
        let forced = read(&self.thinking_forced_closes);
        if forced > 0 {
            declare(
                &mut out,
                "ignis_thinking_forced_closes_total",
                "counter",
                "Completed requests whose reasoning block the thinking budget closed.",
            );
            let _ = writeln!(out, "ignis_thinking_forced_closes_total {forced}");
        }
        // GitHub #330: the loop guard, absent until a request it watched
        // ended or it saw a loop (a load that runs it `off` exports nothing
        // of it), then the whole family: its zeros are what a quiet guard
        // reads as.
        let watched = read(&self.loop_guard_watched);
        if watched + self.loop_guard_loops.iter().map(read).sum::<u64>() > 0 {
            declare(&mut out, "ignis_loop_guard_watched_requests_total", "counter", "Requests the loop guard watched.");
            let _ = writeln!(out, "ignis_loop_guard_watched_requests_total {watched}");
            declare(
                &mut out,
                "ignis_loop_guard_looped_requests_total",
                "counter",
                "Watched requests with at least one loop.",
            );
            let _ = writeln!(out, "ignis_loop_guard_looped_requests_total {}", read(&self.loop_guard_looped));
            declare(
                &mut out,
                "ignis_loop_guard_loops_total",
                "counter",
                "Loops the guard saw, by what it did about them.",
            );
            for (outcome, series) in LoopOutcome::ALL.iter().zip(&self.loop_guard_loops) {
                let _ = writeln!(out, "ignis_loop_guard_loops_total{{outcome=\"{}\"}} {}", outcome.label(), read(series));
            }
            declare(
                &mut out,
                "ignis_loop_guard_rewound_tokens_total",
                "counter",
                "Generated tokens the guard's rewinds took back.",
            );
            let _ = writeln!(out, "ignis_loop_guard_rewound_tokens_total {}", read(&self.loop_guard_rewound_tokens));
            self.loop_guard_min_ratio.render(
                &mut out,
                "ignis_loop_guard_min_ratio",
                "Lowest loop score each watched request reached.",
            );
        }
        // GitHub #307: speculative decoding, absent until a request that ran
        // verify rounds ends (a load without speculation exports nothing of
        // it). Acceptance is accepted / drafted; per position, a draft at
        // position p counts in `drafted` when a round proposed it and in
        // `accepted` when the round committed it too.
        let rounds = read(&self.spec_rounds);
        if rounds > 0 {
            for (name, help, series) in [
                ("ignis_speculative_rounds_total", "Verify rounds run.", &self.spec_rounds),
                ("ignis_speculative_drafted_tokens_total", "Draft tokens verified.", &self.spec_drafted),
                ("ignis_speculative_accepted_tokens_total", "Draft tokens committed.", &self.spec_accepted),
            ] {
                declare(&mut out, name, "counter", help);
                let _ = writeln!(out, "{name} {}", read(series));
            }
            for (name, help, family) in [
                (
                    "ignis_speculative_position_drafted_total",
                    "Verify rounds that proposed a draft at the position.",
                    &self.spec_drafted_at,
                ),
                (
                    "ignis_speculative_position_accepted_total",
                    "Verify rounds that committed the draft at the position.",
                    &self.spec_accepted_at,
                ),
            ] {
                declare(&mut out, name, "counter", help);
                for (j, series) in family.iter().enumerate() {
                    let _ = writeln!(out, "{name}{{position=\"{}\"}} {}", j + 1, read(series));
                }
            }
        }
        // GitHub #285: absent until the first call, for the decision
        // family's reason.
        if self.tokenize_calls.iter().map(read).sum::<u64>() > 0 {
            declare(
                &mut out,
                "ignis_tokenize_requests_total",
                "counter",
                "Calls to the counting routes, which never enter the scheduler, by route.",
            );
            for (route, series) in TokenizeRoute::ALL.iter().zip(&self.tokenize_calls) {
                let _ = writeln!(out, "ignis_tokenize_requests_total{{route=\"{}\"}} {}", route.label(), read(series));
            }
        }
        // GitHub #324: absent until an image job was admitted or ended, as
        // the counting routes' counter is.
        // GitHub #301, #302: last, so a 27B load's exposition is the same
        // bytes it was before Flash-Next.
        out
    }
}

/// A metric's `HELP` and `TYPE` lines.
fn declare(out: &mut String, name: &str, kind: &str, help: &str) {
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} {kind}");
}

/// A label value escaped for the text format: backslash, double quote and
/// line feed.
fn escape_label_value(value: &str) -> String {
    value.replace('\\', r"\\").replace('"', "\\\"").replace('\n', r"\n")
}

/// `GET path` over `metrics`: `/metrics` for the metrics listener,
/// `/ui/metrics` for the Playground's copy on the API listener.
pub fn router<S>(path: &str, metrics: Arc<Metrics>) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    Router::new().route(
        path,
        get(move || async move { ([(header::CONTENT_TYPE, CONTENT_TYPE)], metrics.render()) }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sample lines of `text`: `(name, labels, value)`, comments skipped.
    fn samples(text: &str) -> Vec<(String, String, String)> {
        text.lines()
            .filter(|line| !line.starts_with('#') && !line.is_empty())
            .map(|line| {
                let (series, value) = line.rsplit_once(' ').expect("`series value`");
                let (name, labels) = match series.split_once('{') {
                    Some((name, labels)) => (name, labels.trim_end_matches('}')),
                    None => (series, ""),
                };
                (name.to_owned(), labels.to_owned(), value.to_owned())
            })
            .collect()
    }

    fn value(text: &str, name: &str, labels: &str) -> String {
        samples(text)
            .into_iter()
            .find(|(n, l, _)| n == name && l == labels)
            .unwrap_or_else(|| panic!("no `{name}{{{labels}}}` in:\n{text}"))
            .2
    }

    #[test]
    fn every_metric_is_declared_once_with_help_and_type_before_its_samples() {
        declared_once(&Metrics::new().render(), &[]);
        // And the forced-close counter (spec server/08), on the same terms.
        let forced = Metrics::new();
        forced.record_thinking_forced_close();
        declared_once(&forced.render(), &[("ignis_thinking_forced_closes_total", "counter")]);
        // And the loop guard's family (GitHub #330), once it watched a request.
        let guarded = Metrics::new();
        guarded.record_loop_guard_watched();
        declared_once(
            &guarded.render(),
            &[
                ("ignis_loop_guard_watched_requests_total", "counter"),
                ("ignis_loop_guard_looped_requests_total", "counter"),
                ("ignis_loop_guard_loops_total", "counter"),
                ("ignis_loop_guard_rewound_tokens_total", "counter"),
                ("ignis_loop_guard_min_ratio", "histogram"),
            ],
        );
        // And the counting routes' counter (GitHub #285).
        let counted = Metrics::new();
        counted.record_tokenize(TokenizeRoute::Tokenize);
        declared_once(&counted.render(), &[("ignis_tokenize_requests_total", "counter")]);

    }

    /// GitHub #285: absent until a counting route is called, then both routes
    /// have a series and a call moves exactly its own.
    #[test]
    fn the_tokenize_counter_is_absent_until_a_call_and_counts_by_route() {
        let metrics = Metrics::new();
        assert!(!metrics.render().contains("ignis_tokenize_requests_total"));
        metrics.record_tokenize(TokenizeRoute::Tokenize);
        metrics.record_tokenize(TokenizeRoute::Tokenize);
        let text = metrics.render();
        assert_eq!(value(&text, "ignis_tokenize_requests_total", "route=\"tokenize\""), "2", "{text}");
        assert_eq!(value(&text, "ignis_tokenize_requests_total", "route=\"detokenize\""), "0", "{text}");
    }

    /// GitHub #307: the speculative family is absent until a request that
    /// ran verify rounds ends, then sums the requests' counters.
    #[test]
    fn the_speculative_counters_sum_the_requests_verify_rounds() {
        let metrics = Metrics::new();
        assert!(!metrics.render().contains("ignis_speculative_rounds_total"));
        let spec = ignis_core::types::SpecCounters::round(3, 2) + ignis_core::types::SpecCounters::round(3, 0);
        metrics.record_speculation(&spec);
        metrics.record_speculation(&ignis_core::types::SpecCounters::round(1, 1));
        let text = metrics.render();
        declared_once(
            &text,
            &[
                ("ignis_speculative_rounds_total", "counter"),
                ("ignis_speculative_drafted_tokens_total", "counter"),
                ("ignis_speculative_accepted_tokens_total", "counter"),
                ("ignis_speculative_position_drafted_total", "counter"),
                ("ignis_speculative_position_accepted_total", "counter"),
            ],
        );
        assert_eq!(value(&text, "ignis_speculative_rounds_total", ""), "3");
        assert_eq!(value(&text, "ignis_speculative_drafted_tokens_total", ""), "7");
        assert_eq!(value(&text, "ignis_speculative_accepted_tokens_total", ""), "3");
        assert_eq!(value(&text, "ignis_speculative_position_drafted_total", "position=\"1\""), "3");
        assert_eq!(value(&text, "ignis_speculative_position_accepted_total", "position=\"1\""), "2");
        assert_eq!(value(&text, "ignis_speculative_position_accepted_total", "position=\"2\""), "1");
        assert_eq!(value(&text, "ignis_speculative_position_drafted_total", "position=\"3\""), "2");
    }

    /// A load's reservations with nothing in them but KV-disk's budget.
    fn with_disk(kv_disk_bytes: Option<u64>) -> LoadReservations {
        LoadReservations {
            lines: ignis_core::VramLines::default(),
            budget_bytes: 0,
            kv_pool_pages: 0,
            kv_page_bytes: 0,
            kv_ram_arena_bytes: 0,
            retained_slots: 0,
            retained_host_slots: 0,
            retained_host_bytes: 0,
            kv_disk_bytes,
            }
    }

    /// Spec vram-budget/03 AC 38: on a load with KV-disk, its three families
    /// and the retained families' `tier="disk"` rows render from the first
    /// scrape, zeros included; on a load without it, none of them ever does.
    #[test]
    fn the_disk_tier_renders_from_the_first_scrape_only_on_a_load_that_has_it() {
        let without = Metrics::new();
        without.set_load_reservations(with_disk(None));
        let text = without.render();
        assert!(!text.contains("ignis_kv_disk_"), "{text}");
        assert!(!text.contains("tier=\"disk\""), "{text}");

        let with = Metrics::new();
        with.set_load_reservations(with_disk(Some(16 << 30)));
        let text = with.render();
        declared_once(
            &text,
            &[
                ("ignis_kv_disk_bytes", "gauge"),
                ("ignis_kv_disk_spills_total", "counter"),
                ("ignis_kv_disk_failures_total", "counter"),
            ],
        );
        assert_eq!(value(&text, "ignis_kv_disk_bytes", "state=\"capacity\""), "17179869184");
        assert_eq!(value(&text, "ignis_kv_disk_bytes", "state=\"used\""), "0");
        for from in ["device", "kv_ram"] {
            assert_eq!(value(&text, "ignis_kv_disk_spills_total", &format!("from=\"{from}\"")), "0");
        }
        for op in ["write", "read"] {
            assert_eq!(value(&text, "ignis_kv_disk_failures_total", &format!("op=\"{op}\"")), "0");
        }
        for name in [
            "ignis_retained_reused_tokens_total",
            "ignis_retained_state_hits_total",
            "ignis_retained_state_misses_total",
            "ignis_retained_state_spills_total",
            "ignis_retained_state_discards_total",
            "ignis_retained_state_restores_total",
        ] {
            for kind in ["checkpoint", "prefix"] {
                assert_eq!(value(&text, name, &format!("tier=\"disk\",kind=\"{kind}\"")), "0", "{name}");
            }
        }

        with.record_retained_state(RetainedStateOperation::Spill, ReuseSource::Disk, RetainedKind::Checkpoint);
        with.record_retained_reused(ReuseSource::Disk, RetainedKind::Checkpoint, 9000);
        with.set_occupancy(ignis_core::Occupancy {
            kv_used_pages: 0,
            kv_pool_pages: 0,
            kv_ram_used_bytes: 0,
            kv_disk_used_bytes: 5 << 30,
        });
        let text = with.render();
        assert_eq!(value(&text, "ignis_retained_state_spills_total", "tier=\"disk\",kind=\"checkpoint\""), "1");
        assert_eq!(value(&text, "ignis_retained_reused_tokens_total", "tier=\"disk\",kind=\"checkpoint\""), "9000");
        assert_eq!(value(&text, "ignis_kv_disk_bytes", "state=\"used\""), "5368709120");
    }

    #[test]
    fn the_forced_close_counter_is_absent_until_a_close_is_forced() {
        let metrics = Metrics::new();
        assert!(!metrics.render().contains("ignis_thinking_forced_closes_total"));
        metrics.record_thinking_forced_close();
        metrics.record_thinking_forced_close();
        assert_eq!(value(&metrics.render(), "ignis_thinking_forced_closes_total", ""), "2");
    }

    /// GitHub #330: nothing of the loop guard until it watched a request,
    /// then the whole family, zeros included: "no loop in N watched turns"
    /// is the reading a quiet guard gives.
    #[test]
    fn the_loop_guard_family_is_absent_until_a_request_is_watched_then_whole() {
        let metrics = Metrics::new();
        assert!(!metrics.render().contains("ignis_loop_guard_"));
        metrics.record_loop_guard_watched();
        let text = metrics.render();
        assert_eq!(value(&text, "ignis_loop_guard_watched_requests_total", ""), "1", "{text}");
        assert_eq!(value(&text, "ignis_loop_guard_looped_requests_total", ""), "0", "{text}");
        assert_eq!(value(&text, "ignis_loop_guard_rewound_tokens_total", ""), "0", "{text}");
        assert_eq!(value(&text, "ignis_loop_guard_min_ratio_count", ""), "0", "{text}");
        for outcome in LoopOutcome::ALL {
            assert_eq!(value(&text, "ignis_loop_guard_loops_total", &format!("outcome=\"{}\"", outcome.label())), "0", "{text}");
        }
        let labels: Vec<_> = LoopOutcome::ALL.iter().map(|outcome| outcome.label()).collect();
        assert_eq!(labels, ["rewound", "closed", "detected", "too_little_kept", "no_room", "no_tokenizer", "submit_refused"]);
        // A loop in the first watched request, before it ended: the family
        // is there, its request not counted yet.
        let looping = Metrics::new();
        looping.record_loop_guard_loop(LoopOutcome::Rewound);
        let text = looping.render();
        assert_eq!(value(&text, "ignis_loop_guard_loops_total", "outcome=\"rewound\""), "1", "{text}");
        assert_eq!(value(&text, "ignis_loop_guard_watched_requests_total", ""), "0", "{text}");
    }

    /// GitHub #330: each loop by its outcome, a request that looped once
    /// however often it looped, the tokens rewinds took back, and a
    /// request's lowest score in the margin histogram.
    #[test]
    fn the_loop_guard_counts_loops_by_outcome_looped_requests_tokens_and_margins() {
        let metrics = Metrics::new();
        metrics.record_loop_guard_watched();
        metrics.record_loop_guard_watched();
        metrics.record_loop_guard_looped();
        metrics.record_loop_guard_loop(LoopOutcome::Rewound);
        metrics.record_loop_guard_loop(LoopOutcome::Rewound);
        metrics.record_loop_guard_loop(LoopOutcome::Closed);
        metrics.record_loop_guard_loop(LoopOutcome::TooLittleKept);
        metrics.record_loop_guard_rewound_tokens(400);
        metrics.record_loop_guard_rewound_tokens(125);
        metrics.observe_loop_guard_min_ratio(0.182);
        metrics.observe_loop_guard_min_ratio(0.47);
        let text = metrics.render();
        assert_eq!(value(&text, "ignis_loop_guard_watched_requests_total", ""), "2", "{text}");
        assert_eq!(value(&text, "ignis_loop_guard_looped_requests_total", ""), "1", "{text}");
        assert_eq!(value(&text, "ignis_loop_guard_loops_total", "outcome=\"rewound\""), "2", "{text}");
        assert_eq!(value(&text, "ignis_loop_guard_loops_total", "outcome=\"closed\""), "1", "{text}");
        assert_eq!(value(&text, "ignis_loop_guard_loops_total", "outcome=\"too_little_kept\""), "1", "{text}");
        assert_eq!(value(&text, "ignis_loop_guard_loops_total", "outcome=\"detected\""), "0", "{text}");
        assert_eq!(value(&text, "ignis_loop_guard_rewound_tokens_total", ""), "525", "{text}");
        assert_eq!(value(&text, "ignis_loop_guard_min_ratio_bucket", "le=\"0.15\""), "0", "{text}");
        assert_eq!(value(&text, "ignis_loop_guard_min_ratio_bucket", "le=\"0.2\""), "1", "{text}");
        assert_eq!(value(&text, "ignis_loop_guard_min_ratio_bucket", "le=\"0.45\""), "1", "{text}");
        assert_eq!(value(&text, "ignis_loop_guard_min_ratio_bucket", "le=\"0.5\""), "2", "{text}");
        assert_eq!(value(&text, "ignis_loop_guard_min_ratio_count", ""), "2", "{text}");
        assert_eq!(value(&text, "ignis_loop_guard_min_ratio_sum", ""), "0.652", "{text}");
    }

    /// Every metric in the base contract, plus `also`, declared once with
    /// `HELP` and `TYPE` before its first sample.
    fn declared_once(text: &str, also: &[(&str, &str)]) {
        let expected = [
            ("ignis_build_info", "gauge"),
            ("ignis_scheduler_requests", "gauge"),
            ("ignis_requests_accepted_total", "counter"),
            ("ignis_requests_completed_total", "counter"),
            ("ignis_requests_cancelled_total", "counter"),
            ("ignis_generated_tokens_total", "counter"),
            ("ignis_decoded_tokens_total", "counter"),
            ("ignis_kv_cache_evictions_total", "counter"),
            ("ignis_kv_ram_evictions_total", "counter"),
            ("ignis_prefix_reused_tokens_total", "counter"),
            ("ignis_retained_reused_tokens_total", "counter"),
            ("ignis_retained_state_hits_total", "counter"),
            ("ignis_retained_state_misses_total", "counter"),
            ("ignis_retained_state_spills_total", "counter"),
            ("ignis_retained_state_discards_total", "counter"),
            ("ignis_retained_state_restores_total", "counter"),
            ("ignis_retained_slot_skips_total", "counter"),
            ("ignis_vram_reserved_bytes", "gauge"),
            ("ignis_vram_budget_bytes", "gauge"),
            ("ignis_kv_pool_pages", "gauge"),
            ("ignis_kv_page_bytes", "gauge"),
            ("ignis_kv_pool_used_pages", "gauge"),
            ("ignis_kv_ram_arena_bytes", "gauge"),
            ("ignis_retained_slots", "gauge"),
            ("ignis_retained_host_slots", "gauge"),
            ("ignis_retained_host_bytes", "gauge"),
            ("ignis_responses_sockets", "gauge"),
            ("ignis_responses_queued_requests", "gauge"),
            ("ignis_requests_rejected_total", "counter"),
            ("ignis_request_ttft_seconds", "histogram"),
            ("ignis_request_duration_seconds", "histogram"),
        ];
        let lines: Vec<&str> = text.lines().collect();
        for (name, kind) in expected.iter().chain(also).copied() {
            let help = lines
                .iter()
                .position(|l| l.starts_with(&format!("# HELP {name} ")))
                .unwrap_or_else(|| panic!("no HELP for {name}:\n{text}"));
            assert_eq!(lines[help + 1], format!("# TYPE {name} {kind}"), "{text}");
            let first_sample = lines
                .iter()
                .position(|l| {
                    l.starts_with(&format!("{name} "))
                        || l.starts_with(&format!("{name}{{"))
                        || (kind == "histogram" && l.starts_with(&format!("{name}_bucket{{")))
                })
                .unwrap_or_else(|| panic!("no sample for {name}:\n{text}"));
            assert!(first_sample > help + 1, "{name}'s samples follow its TYPE:\n{text}");
            assert_eq!(
                lines.iter().filter(|l| l.starts_with(&format!("# TYPE {name} "))).count(),
                1,
                "{text}"
            );
        }
        assert!(text.ends_with('\n'), "the exposition ends with a line feed");
    }

    #[test]
    fn a_fresh_projection_reports_build_identity_and_zeros() {
        let text = Metrics::new().render();
        let version = format!("version=\"{}\"", env!("CARGO_PKG_VERSION"));
        assert_eq!(value(&text, "ignis_build_info", &version), "1");
        assert_eq!(value(&text, "ignis_scheduler_requests", "state=\"waiting\""), "0");
        assert_eq!(value(&text, "ignis_scheduler_requests", "state=\"running\""), "0");
        for name in [
            "ignis_requests_accepted_total",
            "ignis_requests_completed_total",
            "ignis_requests_cancelled_total",
            "ignis_generated_tokens_total",
            "ignis_decoded_tokens_total",
            "ignis_kv_cache_evictions_total",
            "ignis_kv_ram_evictions_total",
            "ignis_prefix_reused_tokens_total",
        ] {
            assert_eq!(value(&text, name, ""), "0", "{name}");
        }
        for reason in ["full", "unknown_model", "oversized"] {
            assert_eq!(value(&text, "ignis_requests_rejected_total", &format!("reason=\"{reason}\"")), "0");
        }
    }

    #[test]
    fn a_submit_error_counts_under_its_fixed_reason() {
        assert_eq!(Rejection::of(&SubmitError::Full), Rejection::Full);
        assert_eq!(Rejection::of(&SubmitError::UnknownModel("x".into())), Rejection::UnknownModel);
        assert_eq!(Rejection::of(&SubmitError::Oversized), Rejection::Oversized);
        assert_eq!(
            Rejection::of(&SubmitError::ContextExceeded { requested: 9000, limit: 8192 }),
            Rejection::Oversized
        );
    }

    #[test]
    fn a_scaled_integer_is_rendered_without_trailing_zeros() {
        // Milliseconds as seconds, the latency histograms' unit.
        assert_eq!(decimal(0, THOUSANDTHS), "0");
        assert_eq!(decimal(50, THOUSANDTHS), "0.05");
        assert_eq!(decimal(2_500, THOUSANDTHS), "2.5");
        assert_eq!(decimal(600_000, THOUSANDTHS), "600");
        assert_eq!(decimal(1_001, THOUSANDTHS), "1.001");
        // And millionths as a ratio, which needs the width to come from the
        // scale rather than from a literal: `{frac:03}` would render
        // 998,002 millionths as "0.998002" by luck and 2 as "0.2".
        assert_eq!(decimal(1_000_000, MILLIONTHS), "1");
        assert_eq!(decimal(998_002, MILLIONTHS), "0.998002");
        assert_eq!(decimal(999_500, MILLIONTHS), "0.9995");
        assert_eq!(decimal(2, MILLIONTHS), "0.000002");
        assert_eq!(decimal(3_992_008, MILLIONTHS), "3.992008");
    }

    #[test]
    fn recorded_facts_move_their_series() {
        let metrics = Metrics::new();
        metrics.record_accepted();
        metrics.record_accepted();
        metrics.record_completed(7);
        metrics.record_completed(5);
        metrics.record_cancelled();
        metrics.record_decoded_token();
        metrics.record_decoded_token();
        metrics.set_scheduler_requests(3, 4);
        metrics.set_scheduler_requests(1, 2);

        let text = metrics.render();
        assert_eq!(value(&text, "ignis_requests_accepted_total", ""), "2");
        assert_eq!(value(&text, "ignis_requests_completed_total", ""), "2");
        assert_eq!(value(&text, "ignis_requests_cancelled_total", ""), "1");
        assert_eq!(value(&text, "ignis_generated_tokens_total", ""), "12");
        // Decoded tokens are their own series, not derived from completions.
        assert_eq!(value(&text, "ignis_decoded_tokens_total", ""), "2");
        // Gauges are the latest state, not a sum.
        assert_eq!(value(&text, "ignis_scheduler_requests", "state=\"waiting\""), "1");
        assert_eq!(value(&text, "ignis_scheduler_requests", "state=\"running\""), "2");
    }

    #[test]
    fn the_operational_counters_move_with_their_facts() {
        let metrics = Metrics::new();
        metrics.record_eviction();
        metrics.record_eviction();
        metrics.record_kv_ram_eviction();
        metrics.record_prefix_reused(32);
        metrics.record_prefix_reused(64);
        metrics.record_rejected(Rejection::Full);
        metrics.record_rejected(Rejection::Oversized);
        metrics.record_rejected(Rejection::Oversized);

        let text = metrics.render();
        assert_eq!(value(&text, "ignis_kv_cache_evictions_total", ""), "2");
        // The two tiers' evictions never bleed into each other: two
        // snapshots arrived at KV-RAM, one live snapshot left it (#224).
        assert_eq!(value(&text, "ignis_kv_ram_evictions_total", ""), "1");
        assert_eq!(value(&text, "ignis_prefix_reused_tokens_total", ""), "96");
        assert_eq!(value(&text, "ignis_requests_rejected_total", "reason=\"full\""), "1");
        assert_eq!(value(&text, "ignis_requests_rejected_total", "reason=\"unknown_model\""), "0");
        assert_eq!(value(&text, "ignis_requests_rejected_total", "reason=\"oversized\""), "2");
    }

    #[test]
    fn retained_state_operations_are_counted_per_residency_tier_and_kind() {
        let metrics = Metrics::new();
        let checkpoint = RetainedKind::Checkpoint;
        metrics.record_retained_state(RetainedStateOperation::Hit, ReuseSource::Device, checkpoint);
        metrics.record_retained_state(RetainedStateOperation::Miss, ReuseSource::Device, checkpoint);
        metrics.record_retained_state(RetainedStateOperation::Spill, ReuseSource::KvRam, checkpoint);
        metrics.record_retained_state(RetainedStateOperation::Discard, ReuseSource::KvRam, checkpoint);
        metrics.record_retained_state(RetainedStateOperation::Restore, ReuseSource::KvRam, checkpoint);

        let text = metrics.render();
        for (name, tier) in [
            ("ignis_retained_state_hits_total", "device"),
            ("ignis_retained_state_misses_total", "device"),
            ("ignis_retained_state_spills_total", "kv_ram"),
            ("ignis_retained_state_discards_total", "kv_ram"),
            ("ignis_retained_state_restores_total", "kv_ram"),
        ] {
            assert_eq!(
                value(&text, name, &format!("tier=\"{tier}\",kind=\"checkpoint\"")),
                "1",
                "{name}"
            );
            // The other kind is its own series and did not move with it —
            // the whole point of the split (GitHub #216).
            assert_eq!(
                value(&text, name, &format!("tier=\"{tier}\",kind=\"prefix\"")),
                "0",
                "{name}"
            );
        }
        assert_eq!(
            value(&text, "ignis_retained_state_hits_total", "tier=\"kv_ram\",kind=\"checkpoint\""),
            "0"
        );
    }

    #[test]
    fn a_prefix_and_a_checkpoint_in_the_same_tier_are_never_summed_into_one_series() {
        let metrics = Metrics::new();
        metrics.record_retained_state(
            RetainedStateOperation::Spill,
            ReuseSource::KvRam,
            RetainedKind::Checkpoint,
        );
        for _ in 0..3 {
            metrics.record_retained_state(
                RetainedStateOperation::Spill,
                ReuseSource::KvRam,
                RetainedKind::Prefix,
            );
        }
        metrics.record_retained_reused(ReuseSource::Device, RetainedKind::Prefix, 64);
        metrics.record_retained_reused(ReuseSource::KvRam, RetainedKind::Checkpoint, 1536);

        let text = metrics.render();
        let spills = |kind: &str| {
            value(&text, "ignis_retained_state_spills_total", &format!("tier=\"kv_ram\",kind=\"{kind}\""))
        };
        assert_eq!(spills("checkpoint"), "1");
        assert_eq!(spills("prefix"), "3", "four spills, and the load shed prefixes");
        assert_eq!(
            value(&text, "ignis_retained_reused_tokens_total", "tier=\"device\",kind=\"prefix\""),
            "64"
        );
        assert_eq!(
            value(&text, "ignis_retained_reused_tokens_total", "tier=\"kv_ram\",kind=\"checkpoint\""),
            "1536"
        );
        // Sibling-prefix reuse keeps its own meaning and takes no kind.
        assert_eq!(value(&text, "ignis_prefix_reused_tokens_total", ""), "0");
    }

    #[test]
    fn the_reservations_are_exported_with_the_plan_own_line_spellings() {
        let metrics = Metrics::new();
        // A fresh projection reserves nothing and holds nothing.
        let empty = metrics.render();
        for (line, _) in ignis_core::VramLines::default().entries() {
            assert_eq!(value(&empty, "ignis_vram_reserved_bytes", &format!("line=\"{line}\"")), "0");
        }
        assert_eq!(value(&empty, "ignis_kv_pool_pages", ""), "0");

        let mut lines = ignis_core::VramLines::default();
        lines.weights = 21_000_000_000;
        lines.cuda_context = 600_000_000;
        lines.retained_slots = 1_000_000_000;
        metrics.set_load_reservations(LoadReservations {
            lines,
            budget_bytes: 30_000_000_000,
            kv_pool_pages: 5_000,
            kv_page_bytes: 1_048_576,
            kv_ram_arena_bytes: 8 << 30,
            retained_slots: 9,
            retained_host_slots: 7,
            retained_host_bytes: 7 * 232_532_224,
            kv_disk_bytes: None,
            });

        let text = metrics.render();
        // The thirteen lines are the plan's, in the plan's order and spelling,
        // then Flash-Next's two (GitHub #306) and the image model's condition
        // budget (GitHub #324).
        let exported: Vec<String> = samples(&text)
            .into_iter()
            .filter(|(name, _, _)| name == "ignis_vram_reserved_bytes")
            .map(|(_, labels, _)| labels)
            .collect();
        let expected: Vec<String> = ignis_core::VramLines::default()
            .entries()
            .iter()
            .map(|(line, _)| format!("line=\"{line}\""))
            .collect();
        assert_eq!(exported, expected, "{text}");
        assert_eq!(value(&text, "ignis_vram_reserved_bytes", "line=\"weights\""), "21000000000");
        assert_eq!(value(&text, "ignis_vram_reserved_bytes", "line=\"retained_slots\""), "1000000000");
        assert_eq!(value(&text, "ignis_vram_budget_bytes", ""), "30000000000");
        assert_eq!(value(&text, "ignis_kv_pool_pages", ""), "5000");
        assert_eq!(value(&text, "ignis_kv_page_bytes", ""), "1048576");
        assert_eq!(value(&text, "ignis_kv_ram_arena_bytes", "state=\"capacity\""), "8589934592");
        assert_eq!(value(&text, "ignis_retained_slots", "state=\"capacity\""), "9");
        // GitHub #281: of the nine, the host ones and their pinned block --
        // host memory, beside the VRAM lines rather than one of them.
        assert_eq!(value(&text, "ignis_retained_host_slots", ""), "7");
        assert_eq!(value(&text, "ignis_retained_host_bytes", ""), "1627725568");
        // Nothing is occupied until a step reports occupancy.
        assert_eq!(value(&text, "ignis_kv_pool_used_pages", ""), "0");
        assert_eq!(value(&text, "ignis_kv_ram_arena_bytes", "state=\"used\""), "0");
        assert_eq!(value(&text, "ignis_retained_slots", "state=\"in_use\""), "0");
    }

    #[test]
    fn occupancy_and_slot_skips_follow_the_facts_and_come_back_to_zero() {
        let metrics = Metrics::new();
        metrics.set_occupancy(ignis_core::Occupancy {
            kv_used_pages: 412,
            kv_pool_pages: 1_000,
            kv_ram_used_bytes: 3 << 30,
                kv_disk_used_bytes: 0,
});
        metrics.set_retained_slots_in_use(4);
        metrics.record_retained_slot_skip(RetainedSkip::PublishNoSlot);
        metrics.record_retained_slot_skip(RetainedSkip::CaptureNoPage);
        metrics.record_retained_slot_skip(RetainedSkip::CaptureNoPage);

        let text = metrics.render();
        assert_eq!(value(&text, "ignis_kv_pool_used_pages", ""), "412");
        assert_eq!(value(&text, "ignis_kv_ram_arena_bytes", "state=\"used\""), "3221225472");
        assert_eq!(value(&text, "ignis_retained_slots", "state=\"in_use\""), "4");
        for (reason, count) in [
            ("publish_skipped_no_slot", "1"),
            ("capture_skipped_no_slot", "0"),
            ("capture_skipped_no_page", "2"),
        ] {
            assert_eq!(
                value(&text, "ignis_retained_slot_skips_total", &format!("reason=\"{reason}\"")),
                count
            );
        }

        // The last step of a load releases everything, and reports it: these
        // are gauges of the latest reading, not high-water marks.
        metrics.set_occupancy(ignis_core::Occupancy { kv_pool_pages: 1_000, ..Default::default() });
        metrics.set_retained_slots_in_use(0);
        let text = metrics.render();
        assert_eq!(value(&text, "ignis_kv_pool_used_pages", ""), "0");
        assert_eq!(value(&text, "ignis_kv_ram_arena_bytes", "state=\"used\""), "0");
        assert_eq!(value(&text, "ignis_retained_slots", "state=\"in_use\""), "0");
        // A counter does not come back: the skips already happened.
        assert_eq!(
            value(&text, "ignis_retained_slot_skips_total", "reason=\"capture_skipped_no_page\""),
            "2"
        );
    }

    #[test]
    fn no_exported_series_is_a_percentage() {
        // ADR 0030: every memory series is bytes, pages or slots, so a
        // reader always has both terms and never a ratio that hides which
        // of them moved.
        //
        // Both shapes of exposition, because the decision family is absent
        // from a fresh one (GitHub #241) and it is the one series here that
        // *is* a ratio — answer mass has no second term, which is the ADR's
        // own distinction and worth having under a test.
        let served = Metrics::new();
        for text in [Metrics::new().render(), served.render()] {
            for line in text.lines().filter(|l| l.starts_with("# HELP ignis_")) {
                let name = line.split_whitespace().nth(2).expect("# HELP <name> <help>");
                assert!(
                    !name.contains("_pct") && !name.contains("percent") && !name.contains("_ratio"),
                    "{name} is a percentage"
                );
            }
        }
    }

    #[test]
    fn confidence_is_never_exported() {
        // Spec 05 states it as a rule rather than an omission: confidence is
        // per-caller and per-domain, and a histogram over callers who each
        // mean something different by it means nothing. Pinned, so the
        // omission cannot be undone by somebody adding "the other number
        // the endpoint already has".
        let metrics = Metrics::new();
        let text = metrics.render();
        assert!(!text.contains("confidence"), "{text}");
    }

    /// ADR 0017's fixed boundaries, in seconds, `+Inf` implied.
    const TTFT_BOUNDS: [&str; 12] =
        ["0.05", "0.1", "0.25", "0.5", "1", "2.5", "5", "10", "30", "60", "120", "300"];
    const DURATION_BOUNDS: [&str; 12] =
        ["0.1", "0.25", "0.5", "1", "2.5", "5", "10", "30", "60", "120", "300", "600"];

    /// The `le` values of `name`'s buckets, in exposition order.
    fn bucket_bounds(text: &str, name: &str) -> Vec<String> {
        samples(text)
            .into_iter()
            .filter(|(n, _, _)| *n == format!("{name}_bucket"))
            .map(|(_, labels, _)| {
                labels.strip_prefix("le=\"").and_then(|l| l.strip_suffix('"')).expect("only `le`").to_owned()
            })
            .collect()
    }

    /// The amended contract's mass boundaries, `+Inf` implied.
    const MASS_BOUNDS: [&str; 10] =
        ["0.5", "0.9", "0.95", "0.98", "0.99", "0.995", "0.998", "0.999", "0.9995", "1"];

    /// The loop guard's margin boundaries (GitHub #330), `+Inf` implied.
    const MIN_RATIO_BOUNDS: [&str; 10] = ["0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5", "1"];

    #[test]
    fn the_histograms_use_exactly_the_adr_s_buckets_and_positive_infinity() {
        let text = Metrics::new().render();
        for (name, bounds) in [
            ("ignis_request_ttft_seconds", TTFT_BOUNDS),
            ("ignis_request_duration_seconds", DURATION_BOUNDS),
        ] {
            let mut expected: Vec<String> = bounds.iter().map(|b| (*b).to_owned()).collect();
            expected.push("+Inf".to_owned());
            assert_eq!(bucket_bounds(&text, name), expected, "{text}");
            assert_eq!(value(&text, &format!("{name}_sum"), ""), "0");
            assert_eq!(value(&text, &format!("{name}_count"), ""), "0");
        }
        // The loop guard's margin (GitHub #330), once a request was watched.
        let metrics = Metrics::new();
        metrics.record_loop_guard_watched();
        let text = metrics.render();
        let mut expected: Vec<String> = MIN_RATIO_BOUNDS.iter().map(|b| (*b).to_owned()).collect();
        expected.push("+Inf".to_owned());
        assert_eq!(bucket_bounds(&text, "ignis_loop_guard_min_ratio"), expected, "{text}");
    }

    #[test]
    fn an_observation_lands_in_every_bucket_at_or_above_it() {
        let metrics = Metrics::new();
        // 50 ms sits on the first TTFT boundary (`le` is inclusive); 250 ms
        // on the third; 400 s is past the last one, so only `+Inf` has it.
        metrics.observe_ttft_ms(50);
        metrics.observe_ttft_ms(250);
        metrics.observe_ttft_ms(400_000);
        metrics.observe_duration_ms(700);

        let text = metrics.render();
        let ttft = |le: &str| value(&text, "ignis_request_ttft_seconds_bucket", &format!("le=\"{le}\""));
        assert_eq!(ttft("0.05"), "1");
        assert_eq!(ttft("0.1"), "1");
        assert_eq!(ttft("0.25"), "2");
        assert_eq!(ttft("300"), "2");
        assert_eq!(ttft("+Inf"), "3");
        assert_eq!(value(&text, "ignis_request_ttft_seconds_count", ""), "3");
        assert_eq!(value(&text, "ignis_request_ttft_seconds_sum", ""), "400.3");

        let duration = |le: &str| value(&text, "ignis_request_duration_seconds_bucket", &format!("le=\"{le}\""));
        assert_eq!(duration("0.5"), "0");
        assert_eq!(duration("1"), "1");
        assert_eq!(duration("+Inf"), "1");
        assert_eq!(value(&text, "ignis_request_duration_seconds_sum", ""), "0.7");
    }

    #[test]
    fn a_label_value_is_escaped_per_the_text_format() {
        assert_eq!(escape_label_value(r#"a"b\c"#), r#"a\"b\\c"#);
        assert_eq!(escape_label_value("a\nb"), r"a\nb");
    }

    }
