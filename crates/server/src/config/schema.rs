//! Every configurable field, each declared once (ADR 0046, spec
//! config-v2/01).
//!
//! One [`config_group!`] invocation per group lists that group's fields; one
//! entry per field expands to both the struct field and its [`FieldMeta`] —
//! the flag, env var, file key, default, validator, applicability, the
//! `GET`/`PATCH /v1/config` attributes and the help text all come from that
//! one written entry, so none of them can drift from the others. One
//! invocation per group rather than per field because `macro_rules!` cannot
//! gather separate invocations into one table; inside it, a field is still
//! written exactly once.
//!
//! A field's entry reads:
//!
//! ```text
//! /// Operator-facing description (rustdoc and `help --fields` alike).
//! name: Kind = default, validator, applicability, [Attr, ...];
//! ```
//!
//! `Kind` is a [`FieldKind`] (`kind.rs`) — it decides the Rust type, the
//! grammar and the written form; the attributes ([`Attr`]) are all off
//! unless listed. The field's flag is `--<group>-<name>`, its env var
//! `IGNIS_<GROUP>_<NAME>`, its file key `<group>.<name>`.
//!
//! Names follow today's flags, with the group's own word dropped where it
//! would repeat (`--vram-headroom-bytes`, not `--vram-vram-headroom-bytes`;
//! `--reuse-prompt`, not `--reuse-prompt-reuse`).

use std::collections::BTreeMap;
use std::path::PathBuf;

use ignis_core::compute::ModelFamily;
use ignis_core::ngram_cache::CacheLocation;
use ignis_core::{ExceptionWeights, KvFormat, PrefillAttention, RopeSetting};
use serde_json::{Map, Value};

use super::field::{Applicability, Attr, FieldMeta, Validator};
use super::kind::*;
use super::source::Resolver;
use super::ConfigError;
use crate::instruction::{DeveloperMessagePolicy, SystemMessagePolicy};

/// Expand one group's field list into its settings struct, its `FIELDS`
/// table, and the resolve/render functions the rest of the module reads
/// them through. See the module doc for the entry syntax.
macro_rules! config_group {
    (
        $(#[doc = $gdoc:literal])*
        $Group:ident = $gname:literal {
            $(
                $(#[doc = $doc:literal])*
                $field:ident: $kind:ty = $default:expr, $validator:expr, $applies:expr $(, [$($attr:ident),* $(,)?])?;
            )*
        }
    ) => {
        $(#[doc = $gdoc])*
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub struct $Group {
            $(
                $(#[doc = $doc])*
                pub $field: <$kind as FieldKind>::Value,
            )*
        }

        impl Default for $Group {
            /// Every field at its hardcoded default.
            fn default() -> Self {
                Self { $( $field: $default, )* }
            }
        }

        impl $Group {
            /// The group's name, as the file key and the env var spell it.
            pub const NAME: &'static str = $gname;

            /// One entry per struct field, in declaration order.
            pub const FIELDS: &'static [FieldMeta] = &[
                $(
                    FieldMeta {
                        group: $gname,
                        name: stringify!($field),
                        kind: <$kind as FieldKind>::TAG,
                        nullable: <$kind as FieldKind>::NULLABLE,
                        default: || <$kind as FieldKind>::render(&$default),
                        file_text: <$kind as FieldKind>::file_text,
                        canonical: |raw| <$kind as FieldKind>::parse(raw).map(|value| <$kind as FieldKind>::render(&value)),
                        description: concat!($($doc, "\n",)*),
                        validator: $validator,
                        applies: $applies,
                        visible: Attr::listed(&[$($(Attr::$attr),*)?], Attr::Visible),
                        patchable: Attr::listed(&[$($(Attr::$attr),*)?], Attr::Patchable),
                        reload_required: Attr::listed(&[$($(Attr::$attr),*)?], Attr::ReloadRequired),
                        scoped: Attr::listed(&[$($(Attr::$attr),*)?], Attr::Scoped),
                        switch: <$kind as FieldKind>::SWITCH,
                        repeatable: <$kind as FieldKind>::REPEATABLE,
                    },
                )*
            ];

            /// The entry for the field called `name`.
            fn meta(name: &str) -> &'static FieldMeta {
                Self::FIELDS.iter().find(|meta| meta.name == name).expect("a field the macro declared")
            }

            /// Every field resolved through `resolver`'s sources, in
            /// declaration order (the first refusal wins, so it is the
            /// first field written here that is wrong).
            pub(super) fn resolve(resolver: &Resolver<'_>) -> Result<Self, ConfigError> {
                Ok(Self {
                    $( $field: resolver.field::<$kind>(Self::meta(stringify!($field)), $default)?, )*
                })
            }

            /// Every field as the file format writes it.
            pub fn render(&self) -> Map<String, Value> {
                let mut fields = Map::new();
                $( fields.insert(stringify!($field).to_owned(), <$kind as FieldKind>::render(&self.$field)); )*
                fields
            }
        }
    };
}

// The tests that pin the attributes' closed defaults declare throwaway
// groups with it, beside the fields they test.
#[cfg(test)]
pub(crate) use config_group;

/// Every group, in the order `help --fields`, a written file and the
/// resolver visit them: builds [`Settings`], [`GROUPS`] and their two
/// whole-config functions from one list, as `config_group!` does per field.
macro_rules! settings {
    ($( $(#[doc = $doc:literal])* $group:ident: $Group:ident, )*) => {
        /// The resolved value of every declared field, grouped as the config
        /// file nests them — spec config-v2/01's "resolved Config". What `GET
        /// /v1/config` shows (filtered to [`FieldMeta::visible`]), what
        /// `config generate` writes, and what [`super::Config`] is derived from.
        #[derive(Debug, Clone, PartialEq, Eq, Default)]
        pub struct Settings {
            $( $(#[doc = $doc])* pub $group: $Group, )*
        }

        /// Every group's name and field table, in declaration order.
        pub const GROUPS: &[(&str, &[FieldMeta])] = &[ $( ($Group::NAME, $Group::FIELDS), )* ];

        impl Settings {
            /// Every group resolved through `resolver`.
            pub(super) fn resolve(resolver: &Resolver<'_>) -> Result<Self, ConfigError> {
                Ok(Self { $( $group: $Group::resolve(resolver)?, )* })
            }

            /// Every field, nested by group, as the file format writes it.
            pub fn render(&self) -> Map<String, Value> {
                let mut groups = Map::new();
                $( groups.insert($Group::NAME.to_owned(), Value::Object(self.$group.render())); )*
                groups
            }
        }

        impl serde::Serialize for Settings {
            /// Nested by group, every field as the file format writes it.
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serde::Serialize::serialize(&Value::Object(self.render()), serializer)
            }
        }

        impl<'de> serde::Deserialize<'de> for Settings {
            /// A whole config document, read as a file is — so an unknown
            /// field or a bad value is refused by name — over the hardcoded
            /// defaults.
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let value = <Value as serde::Deserialize>::deserialize(deserializer)?;
                settings_from_document(&value).map_err(serde::de::Error::custom)
            }
        }

        $(
            impl serde::Serialize for $Group {
                /// Every field as the file format writes it.
                fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                    serde::Serialize::serialize(&Value::Object(self.render()), serializer)
                }
            }

            impl<'de> serde::Deserialize<'de> for $Group {
                /// The group's own section of a config document, read as
                /// [`Settings`] reads a whole one.
                fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                    let value = <Value as serde::Deserialize>::deserialize(deserializer)?;
                    let mut document = Map::new();
                    document.insert($Group::NAME.to_owned(), value);
                    settings_from_document(&Value::Object(document))
                        .map(|settings| settings.$group)
                        .map_err(serde::de::Error::custom)
                }
            }
        )*
    };
}

/// A config document read on its own over the hardcoded defaults: through
/// the file reader and the resolver, the one path every value takes, so a
/// group's `Deserialize` cannot read a field differently from a start. The
/// general values only — a family section waits for a family.
fn settings_from_document(value: &Value) -> Result<Settings, String> {
    let document = super::file::read_document(value, "the document").map_err(|e| e.0)?;
    let sources = super::source::Sources { file: document.values, ..Default::default() };
    super::source::resolve_settings(&sources, None, super::source::Fit::Start)
        .map(|resolution| resolution.settings)
        .map_err(|e| e.0)
}

/// Every field of every group, in declaration order.
pub fn all_fields() -> impl Iterator<Item = &'static FieldMeta> {
    GROUPS.iter().flat_map(|(_, fields)| fields.iter())
}

/// The field `<group>.<name>`, if one is declared.
pub fn field(group: &str, name: &str) -> Option<&'static FieldMeta> {
    all_fields().find(|meta| meta.group == group && meta.name == name)
}

const ALL: Applicability = Applicability::AllFamilies;
const QWEN38_ONLY: Applicability = Applicability::Only(&[ModelFamily::Qwen38_27b]);
const NO_RULE: Validator = Validator::None;

const fn between(min: i64, max: i64) -> Validator {
    Validator::Range { min: Some(min), max: Some(max) }
}

const fn at_least(min: i64) -> Validator {
    Validator::Range { min: Some(min), max: None }
}

/// A decimal field's range, in its own units: `decimal(0.0, 4.0)`.
const fn decimal(min: f64, max: f64) -> Validator {
    Validator::DecimalRange { min: (min * 1000.0) as i64, max: (max * 1000.0) as i64 }
}

config_group! {
    /// The HTTP server itself: where it listens, who may call it, what else
    /// it serves beside `/v1`.
    ServerGroup = "server" {
        /// API listen address (`host:port`), using localhost by default. Changing it
        /// requires a restart.
        bind: Text = super::DEFAULT_BIND.to_owned(), NO_RULE, ALL, [Visible];
        /// Key required in `Authorization: Bearer <key>` for authenticated `/v1`
        /// requests. `auto` generates and prints a key at startup. Unset disables
        /// authentication. `GET /v1/config` hides this field.
        api_key: Opt<ApiKeyKind> = None, NO_RULE, ALL;
        /// Serve Prometheus metrics on `metrics_bind` and, when the UI is enabled, at
        /// `/ui/metrics`. Enabled by default; `false` disables both.
        metrics: Bool = super::DEFAULT_METRICS, NO_RULE, ALL, [Visible];
        /// Separate address for the metrics listener, using loopback by default with no
        /// API key. Requires `metrics`. If the default address is occupied, startup
        /// logs a warning and continues without this listener. Failure to bind a custom
        /// address aborts startup.
        metrics_bind: Text = super::DEFAULT_METRICS_BIND.to_owned(), NO_RULE, ALL, [Visible];
        /// Serve the UI at `/ui/`. A binary built without `web/dist` shows build
        /// instructions.
        ui: Bool = super::DEFAULT_UI, NO_RULE, ALL, [Visible];
        /// Deadline in seconds from admission for non-streaming Chat Completions and
        /// HTTP Responses, including media acquisition. `timeout_reply` controls the
        /// timeout response. 0 disables the deadline.
        request_timeout: Secs = super::DEFAULT_REQUEST_TIMEOUT_SECS, between(0, super::MAX_REQUEST_TIMEOUT_SECS as i64), ALL, [Visible, Patchable];
        /// Deadline in seconds from admission for streaming Chat Completions, HTTP
        /// Responses and Responses over WebSocket. `timeout_reply` controls the timeout
        /// response. 0 disables the deadline.
        stream_timeout: Secs = super::DEFAULT_STREAM_TIMEOUT_SECS, between(0, super::MAX_REQUEST_TIMEOUT_SECS as i64), ALL, [Visible, Patchable];
        /// Response on deadline expiry: `partial` returns the generated output with
        /// `finish_reason: "length"`, or an incomplete Response with reason
        /// `max_output_tokens`. `error` returns HTTP 504 `request_timeout` or
        /// terminates the stream with an error.
        timeout_reply: TimeoutReplyKind = crate::deadline::TimeoutReply::Partial, Validator::OneOf(&["partial", "error"]), ALL,
            [Visible, Patchable];
        /// Seconds before closing an idle Responses WebSocket. A socket is idle when no
        /// client frame arrives and no response is running or queued. The server sends
        /// a normal close frame with the reason; the next request opens a new
        /// connection. 0 disables idle closure.
        socket_idle_close: Secs = super::DEFAULT_SOCKET_IDLE_CLOSE_SECS, between(0, super::MAX_SOCKET_IDLE_CLOSE_SECS as i64), ALL,
            [Visible, Patchable];
        /// Where `system` messages go: `merge` joins a leading run into the
        /// system prompt and keeps a later one as its own block in place;
        /// `strict` answers 400 for a system message that is not first.
        system_message_policy: SystemPolicy = SystemMessagePolicy::Merge, Validator::OneOf(&["merge", "strict"]), ALL, [Visible, Patchable];
        /// Where `developer` messages go: `inplace`, `into-system`,
        /// `after-system`, `one-after-system` or `reject`. A leading developer
        /// message is the system prompt except under `reject`.
        developer_message_policy: DeveloperPolicy = DeveloperMessagePolicy::Inplace,
            Validator::OneOf(&["inplace", "into-system", "after-system", "one-after-system", "reject"]), ALL, [Visible, Patchable];
    }
}

config_group! {
    /// The model being served and the shape of its load: context, KV
    /// format, prefill, thinking defaults, the rotary table.
    ModelGroup = "model" {
        /// Model ID exposed by the service. Defaults to `qwen3.8-27b-16g`. An ID naming
        /// a different model family from the artifact is refused.
        id: Opt<Text> = None, NO_RULE, ALL, [Visible];
        /// The .ninfer artifact to load. Unset, the model is looked for under
        /// `download.path`, and fetched when it is not there.
        artifact: Opt<Path> = None, NO_RULE, ALL, [Visible];
        /// Maximum prompt and generation tokens a request may reserve, bounded by the
        /// attention capacity of the selected KV format.
        max_context: Tokens = super::DEFAULT_MAX_CONTEXT, at_least(1), ALL, [Visible, Patchable, ReloadRequired, Scoped];
        /// Output cap for requests that omit `max_tokens`, including reasoning tokens.
        /// The prompt and output must fit within `max_context`. A request's explicit
        /// cap takes precedence. 0 allows output up to the remaining context.
        default_max_tokens: Tokens = super::DEFAULT_MAX_TOKENS, NO_RULE, ALL, [Visible, Patchable, ReloadRequired, Scoped];
        /// KV storage format: `hq-e8-2b` by default, or `bf16` for reference
        /// comparisons.
        kv_format: KvFormatKind = KvFormat::default(), Validator::OneOf(&["bf16", "hq-e8-2b", "hq_e8_2b", "hq"]), ALL,
            [Visible, Patchable, ReloadRequired, Scoped];
        /// Prompt attention precision: `fp8` by default, or `fp16` for FP16
        /// accumulation.
        prefill_attention: PrefillAttentionKind = PrefillAttention::default(), Validator::OneOf(&PrefillAttention::NAMES), QWEN38_ONLY,
            [Visible, Patchable, ReloadRequired];
        /// Device format for the nine BF16 exception tensors: `bf16` uses the
        /// artifact's format; `fp8` converts them at load time to row-scaled FP8 for
        /// tensor-core execution.
        exception_weights: ExceptionWeightsKind = ExceptionWeights::default(), Validator::OneOf(&ExceptionWeights::NAMES), QWEN38_ONLY,
            [Visible, Patchable, ReloadRequired];
        /// The prefill chunk width, in tokens.
        prefill_chunk: Tokens = super::DEFAULT_PREFILL_CHUNK, Validator::MultipleOf(super::PREFILL_CHUNK_ALIGNMENT as u64), ALL,
            [Visible, Patchable, ReloadRequired, Scoped];
        /// Percentage of model time reserved for decode while a prompt prefills.
        /// Defaults to 25. 0 runs one decode round per chunk; 50 splits time evenly.
        decode_share: Opt<Percent> = None, between(0, 99), ALL, [Visible, Patchable, ReloadRequired, Scoped];
        /// Whether to enable reasoning when a request omits the setting.
        enable_thinking: Bool = true, NO_RULE, ALL, [Visible, Patchable];
        /// Reasoning effort for requests that omit it. Unset uses the template's
        /// default.
        reasoning_effort: Opt<Effort> = None, Validator::OneOf(&["none", "minimal", "low", "medium", "high", "xhigh", "max"]), ALL,
            [Visible, Patchable];
        /// The reasoning tokens a request may spend before the model's close
        /// is forced; `off` = no budget. A request's own thinking_budget
        /// overrides it.
        thinking_budget: ThinkingBudget = Some(super::DEFAULT_THINKING_BUDGET), NO_RULE, ALL, [Visible, Patchable];
        /// The loop guard (ADR 0052): `rewind`, a loop in a request's
        /// reasoning (chat completions, `/v1/responses`) is taken back and
        /// the model nudged, then closed after `loop_guard_retries` nudges;
        /// `log`, it is only logged; `off`, no guard, as benches and gates
        /// run for reference parity.
        loop_guard: LoopGuard = crate::loop_guard::LoopGuardMode::Rewind, Validator::OneOf(crate::loop_guard::LoopGuardMode::NAMES), ALL,
            [Visible, Patchable];
        /// A window of scored reasoning whose zlib ratio is under this many
        /// thousandths is a loop.
        loop_guard_ratio: Permille = crate::loop_guard::DEFAULT_RATIO_PERMILLE, between(1, 999), ALL, [Visible, Patchable];
        /// The loop guard's window, in characters of scored reasoning.
        loop_guard_window: Chars = crate::loop_guard::DEFAULT_WINDOW_CHARS, between(500, 100_000), ALL, [Visible, Patchable];
        /// Whether the loop guard scores the reasoning normalized: fenced
        /// and inline code taken out, every run of digits read as one `0`.
        /// `off` scores it as written, which at this window and ratio
        /// flags far more than the raw prototype did (3,000 and 120).
        loop_guard_normalize: Bool = crate::loop_guard::DEFAULT_NORMALIZE, NO_RULE, ALL, [Visible, Patchable];
        /// The loop guard's raw gate, in thousandths: a loop's last
        /// `loop_guard_window` characters as written must score under it
        /// too, which a table of numbers (a template once its digits fold)
        /// does not. 0, no gate.
        loop_guard_raw_gate: Permille = crate::loop_guard::DEFAULT_RAW_GATE_PERMILLE, between(0, 999), ALL, [Visible, Patchable];
        /// A window of scored reasoning at or over this many thousandths is
        /// clean: a rewind goes back to the start of the last one before
        /// the loop (no clean one since the last nudge: to its end).
        loop_guard_clean: Permille = crate::loop_guard::DEFAULT_CLEAN_PERMILLE, between(1, 999), ALL, [Visible, Patchable];
        /// How many characters before that clean window's start a rewind
        /// cuts.
        loop_guard_margin: Chars = crate::loop_guard::DEFAULT_MARGIN_CHARS, between(0, 100_000), ALL, [Visible, Patchable];
        /// The nudges a looping reasoning gets before the guard closes it.
        loop_guard_retries: Retries = crate::loop_guard::DEFAULT_RETRIES, between(0, 10), ALL, [Visible, Patchable];
        /// How many checks in a row must score a loop before the guard
        /// acts: a loop persists, a list or a log transcribed into the
        /// reasoning ends with its content. 1, at the first.
        loop_guard_persist: Checks = crate::loop_guard::DEFAULT_PERSIST_CHECKS, between(1, 100), ALL, [Visible, Patchable];
        /// A rewind that would keep fewer characters of reasoning than this
        /// is only logged, and that run of loop scores is not acted on: a
        /// reasoning that opens in a loop, or in an enumeration scored as
        /// one, would otherwise be dropped whole. 0, always rewind.
        loop_guard_min_kept: Chars = crate::loop_guard::DEFAULT_MIN_KEPT_CHARS, between(0, 100_000), ALL, [Visible, Patchable];
        /// In `rewind` mode, the characters of reasoning held back from the
        /// client: a rewind whose cut falls in them reaches it with no tag
        /// and none of the loop. 0, none; under ~2,000 it hides almost no
        /// rewind.
        loop_guard_holdback: Chars = crate::loop_guard::DEFAULT_HOLDBACK_CHARS, between(0, 100_000), ALL, [Visible, Patchable];
        /// The loop guard scores the window again after this many
        /// characters of new reasoning.
        loop_guard_check_every: Chars = crate::loop_guard::DEFAULT_CHECK_EVERY_CHARS, between(1, 100_000), ALL, [Visible, Patchable];
        /// A rewind's cut moves back to the line break before it when one
        /// is within this many characters. 0, never.
        loop_guard_snap: Chars = crate::loop_guard::DEFAULT_SNAP_CHARS, between(0, 100_000), ALL, [Visible, Patchable];
        /// The least of a request's `max_tokens` a continuation must have
        /// left after the cut and the nudge: with less, the loop is only
        /// logged.
        loop_guard_min_continuation: Tokens = crate::loop_guard::DEFAULT_MIN_CONTINUATION_TOKENS, at_least(1), ALL, [Visible, Patchable];
        /// How long a continuation the engine refuses as full waits for
        /// room, in seconds, before the loop is left as it is; 0 tries
        /// once. A WebSocket request's continuation waits in the socket's
        /// admission queue instead, as the socket's own requests do.
        loop_guard_submit_wait: Secs = crate::loop_guard::DEFAULT_SUBMIT_WAIT_SECS, between(0, 3600), ALL, [Visible, Patchable];
        /// The sentence, in the model's own voice, the loop guard's first
        /// rewind puts after the kept reasoning, on a paragraph of its own.
        loop_guard_nudge: Text = crate::loop_guard::DEFAULT_NUDGE.to_owned(), NO_RULE, ALL, [Visible, Patchable];
        /// The sentence every later rewind puts there, until
        /// `loop_guard_retries` nudges close the reasoning.
        loop_guard_nudge_again: Text = crate::loop_guard::DEFAULT_NUDGE_AGAIN.to_owned(), NO_RULE, ALL, [Visible, Patchable];
        /// Text rotary scaling: `none` uses the linear table;
        /// `yarn:F[,t=..][,bf=..][,bs=..]` rescales the checkpoint's 262,144-position
        /// context. The default `auto` uses `none` up to 262,144 tokens and YaRN with
        /// factor `max_context / 262,144` above it.
        rope_scaling: Rope = RopeSetting::Auto, NO_RULE, ALL, [Visible, Patchable, ReloadRequired, Scoped];
        /// Seconds with no request using the model after which it is unloaded
        /// — VRAM, pinned host memory and KV-RAM given back — while the server
        /// stays up; the next request loads it again and waits for it. 0 never
        /// unloads.
        idle_unload: Secs = super::DEFAULT_MODEL_IDLE_UNLOAD_SECS, between(0, super::MAX_MODEL_IDLE_UNLOAD_SECS as i64), ALL,
            [Visible, Patchable];
        /// Physical memory to leave available at load time, after accounting for token
        /// embeddings, staging, retained host slots and the KV-RAM arena. If these
        /// reservations leave less than the margin, the load is refused and reports
        /// which reservation to reduce.
        host_margin_bytes: Bytes = super::DEFAULT_HOST_MARGIN_BYTES, NO_RULE, ALL, [Visible, Patchable, ReloadRequired, Scoped];
        /// The 27B's adaptation (ADR 0063): when its VRAM plan refuses a
        /// `max_context` the profile or the default gave, the load lowers it
        /// this many tokens at a time, never below this many -- a start whose
        /// plan does not hold one step is refused. A whole number of KV pages
        /// (64 tokens).
        adapt_context_step: Tokens = super::DEFAULT_ADAPT_CONTEXT_STEP, Validator::MultipleOf(ignis_core::KV_PAGE_TOKENS as u64),
            QWEN38_ONLY, [Visible, Patchable, ReloadRequired];
    }
}

config_group! {
    /// The device memory a load may hold, and how the KV pool takes its
    /// share of it.
    VramGroup = "vram" {
        /// KV pool capacity as bytes (K/M/G suffixes) or tokens (`<n>tok`, `<n>Ktok`,
        /// `<n>Mtok`). Unset uses the remaining VRAM budget. The pool must fit one
        /// `max_context` sequence and one page per retained slot.
        kv_pool_bytes: Opt<KvPool> = None, NO_RULE, ALL, [Visible, Patchable, ReloadRequired, Scoped];
        /// What a derived VRAM budget leaves to the desktop and every other
        /// process on the card: the budget is the memory free at start less
        /// this. Not with `budget_bytes`.
        headroom_bytes: Bytes = super::DEFAULT_VRAM_HEADROOM_BYTES, NO_RULE, ALL, [Visible, Patchable, ReloadRequired, Scoped];
        /// The device memory the whole process may hold, weights included;
        /// refused above free memory. Unset, derived from `headroom_bytes`.
        budget_bytes: Opt<Bytes> = None, at_least(1), ALL, [Visible, Patchable, ReloadRequired, Scoped];
        /// Start above free memory, with a warning. Needs `budget_bytes`.
        allow_oversubscription: Bool = false, NO_RULE, ALL, [Visible, Patchable, ReloadRequired, Scoped];
        /// The 27B's adaptation (ADR 0063): at more than one decode lane, a
        /// context the load lowers is kept only when the KV pool holds it and
        /// a spare reservation of this many tokens, room for a second request;
        /// 0, one sequence is enough. At one lane the plan's own floor decides.
        adapt_spare_tokens: Tokens = super::DEFAULT_ADAPT_SPARE_TOKENS, NO_RULE, QWEN38_ONLY, [Visible, Patchable, ReloadRequired];
    }
}

config_group! {
    /// Prompt reuse across requests and the KV-RAM host tier it keeps state
    /// in (the arena is prompt reuse's, not the VRAM pool's).
    ReuseGroup = "reuse" {
        /// Cross-request state reuse. Off, no prompt checkpoint is captured
        /// or claimed, and no prefix is shared unless `retained_device` or
        /// `retained_host` gives slots for it.
        prompt: Bool = super::DEFAULT_PROMPT_REUSE, NO_RULE, ALL, [Visible, Patchable, ReloadRequired, Scoped];
        /// Retained slots in VRAM, reserved in the VRAM plan and handed out
        /// first. Unset, 0. With prompt reuse off, a count shares heads
        /// between live siblings only.
        retained_device: Opt<Slots> = None, NO_RULE, ALL, [Visible, Patchable, ReloadRequired, Scoped];
        /// Retained slots allocated in one pinned host block, about 222 MiB each.
        /// Capturing or restoring a slot copies its state over PCIe. Defaults to two
        /// slots per decode lane, or zero with prompt reuse disabled.
        retained_host: Opt<Slots> = None, NO_RULE, ALL, [Visible, Patchable, ReloadRequired, Scoped];
        /// Idle seconds after which a main-conversation checkpoint in KV-RAM
        /// ranks as a subagent's. Needs prompt reuse on.
        retained_interactive_ttl: Secs = super::DEFAULT_RETAINED_INTERACTIVE_TTL_SECS, NO_RULE, ALL,
            [Visible, Patchable, ReloadRequired, Scoped];
        /// The KV-RAM host tier's budget: page-locked whole at start and held
        /// for the life of the load, so it is RAM the process holds even idle
        /// (and the figure Windows reports as its shared GPU memory). 0
        /// disables the tier.
        kv_host_pool_bytes: Bytes = super::DEFAULT_HOST_POOL_BYTES, NO_RULE, ALL, [Visible, Patchable, ReloadRequired, Scoped];
    }
}

config_group! {
    /// The runtime model switch (spec model-switch/01): how a switch drains,
    /// and which models a request may switch to by naming them.
    SwitchGroup = "switch" {
        /// How long a model switch lets the old model's running requests
        /// finish before it cancels them, in seconds; 0 cancels them at once.
        drain_timeout: Secs = super::DEFAULT_SWITCH_DRAIN_TIMEOUT_SECS, between(0, super::MAX_SWITCH_DRAIN_TIMEOUT_SECS as i64), ALL,
            [Visible, Patchable];
        /// Whether a request whose model names another model `known_models`
        /// lists switches the server to it, then is served on it. Off refuses
        /// it as an unknown model is. An explicit switch is not affected.
        allow_implicit: Bool = super::DEFAULT_ALLOW_MODEL_SWITCH, NO_RULE, ALL, [Visible, Patchable];
        /// The models a request may switch to by naming them, each with the
        /// artifact it loads from: `<id>=<path>` (the flag is repeatable, the
        /// env var takes `;`-separated pairs, a file a map). The model the
        /// server starts on joins them once its load says its id.
        known_models: KnownModels = BTreeMap::new(), NO_RULE, ALL, [Visible, Patchable];
    }
}

config_group! {
    /// Speculative decoding, and the decode lanes.
    SpecGroup = "spec" {
        /// Speculative backend: `dflash2` or `off`. No backend means no speculation.
        backend: Opt<SpecBackend> = None, Validator::OneOf(&["dflash2", "off"]), ALL, [Visible, Patchable, ReloadRequired, Scoped];
        /// Number of tokens DFlash2 proposes per round. Required when `dflash2` is
        /// enabled.
        draft_tokens: Opt<Tokens> = None, between(1, ignis_core::MAX_DRAFT_TOKENS as i64), ALL, [Visible, Patchable, ReloadRequired, Scoped];
        /// Number of sequences decoded at once. Each lane reserves sequence state
        /// (about 233 MB with DFlash2 and hq-e8-2b), round buffers and two default
        /// retained host slots. The base default is eight lanes; the `rtx50-16gb`
        /// profile sets two.
        decode_lanes: Opt<Lanes> = None, between(1, ignis_core::N_DECODE_LANES as i64), ALL, [Visible, Patchable, ReloadRequired, Scoped];
        /// Drafter proposal head: `full` uses the target's output head; `shortlist`
        /// uses the artifact's Q4 head over the 131,072 most frequent tokens and adds
        /// about 357 MB of VRAM. `auto` selects `full`. Requires a speculative backend;
        /// defaults to `full`.
        draft_head: Opt<DraftHead> = None, Validator::OneOf(&["full", "shortlist", "auto"]), ALL, [Visible, Patchable, ReloadRequired, Scoped];
    }
}

config_group! {
    /// The 27B's vision tower.
    VisionGroup = "vision" {
        /// Load the vision tower and reserve its workspace.
        enabled: Bool = false, NO_RULE, QWEN38_ONLY, [Visible, Patchable, ReloadRequired];
        /// Merged vision tokens per request. An image over it is shrunk to
        /// fit, aspect kept; refused only when its aspect ratio cannot fit, or
        /// when several images total over it. Needs vision on.
        max_tokens: Tokens = ignis_core::DEFAULT_VISION_MAX_TOKENS, between(1, ignis_core::VISION_MAX_TOKENS_LIMIT as i64), QWEN38_ONLY,
            [Visible, Patchable, ReloadRequired];
        /// Encoded images kept for reuse, in MiB, where `vision.residency`
        /// puts the pool: VRAM, or pinned host RAM under `host`. Unset, one
        /// envelope-wide embedding; a pool below that is raised to it. Needs
        /// vision on.
        embedding_pool_mib: Opt<Mib> = None, between(1, super::MAX_VISION_EMBEDDING_POOL_MIB as i64), QWEN38_ONLY,
            [Visible, Patchable, ReloadRequired];
        /// Where the vision tower's weights and the embedding pool live:
        /// `device`, in VRAM for the life of the load (today's encode
        /// latency), or `host`, in pinned host RAM, the weights copied to the
        /// GPU for each image encode (~300 MB and the pool less VRAM, so more
        /// KV pool, a few ms more an encode). A `host` start the host RAM
        /// cannot hold is refused, never moved to the device. Needs vision on.
        residency: VisionResidencyKind = ignis_core::VisionResidency::default(), Validator::OneOf(&ignis_core::VisionResidency::NAMES), QWEN38_ONLY,
            [Visible, Patchable, ReloadRequired];
    }
}

config_group! {
    /// How image parts are acquired.
    MediaGroup = "media" {
        /// Fetch image URLs on private, loopback, link-local, multicast and
        /// CGNAT addresses. Off, so an exposed server cannot be used to probe
        /// the operator's LAN. Needs vision on.
        allow_private_network: Bool = false, NO_RULE, QWEN38_ONLY, [Visible, Patchable, ReloadRequired];
        /// Prepared images kept for reuse, in MiB; 0 keeps none. Needs vision
        /// on.
        cache_mib: Mib = super::DEFAULT_MEDIA_CACHE_MIB as u64, between(0, super::MEDIA_CACHE_MIB_LIMIT as i64), QWEN38_ONLY,
            [Visible, Patchable, ReloadRequired];
    }
}

config_group! {
    /// Fetching a missing model.
    DownloadGroup = "download" {
        /// Fetch the model when no artifact is on disk — asked first when
        /// stdin is a terminal, downloaded straight away when it is not. Off
        /// keeps the placeholder template, and refuses POST
        /// /v1/models/download too: the server never uses the network.
        enabled: Bool = super::DEFAULT_MODEL_DOWNLOAD, NO_RULE, ALL, [Visible];
        /// Where a fetched model lands, and where one fetched earlier is
        /// found.
        path: Path = PathBuf::from(super::DEFAULT_MODEL_DOWNLOAD_PATH), NO_RULE, ALL, [Visible];
        /// Where a fetched model's files come from: Hugging Face, or a mirror
        /// answering its {repo}/resolve/{revision}/{file} route (an
        /// Artifactory or Nexus proxy, a static server laid out the same
        /// way). The pins still say whether the bytes are the published ones.
        endpoint: Text = super::DEFAULT_DOWNLOAD_ENDPOINT.to_owned(), NO_RULE, ALL, [Visible];
        /// Sent as `Authorization: Bearer` to the endpoint, never across a
        /// redirect to another host. Unset, HF_TOKEN is sent to huggingface.co
        /// and nothing to a mirror. Never shown by GET /v1/config or `config
        /// print`.
        token: Opt<Token> = None, NO_RULE, ALL;
        /// An operator catalog: a YAML or JSON file of models fetched beside
        /// the built-in ones, in the same format; an id the built-in catalog
        /// has is refused. A relative path is read from the directory of the
        /// config file that names it.
        catalog: Opt<Path> = None, NO_RULE, ALL, [Visible];
        /// Let POST /v1/models/download, the Server view's Download, fetch a
        /// catalog entry while the server runs, under the same API key as
        /// PATCH /v1/config; it needs `enabled` too. Off refuses it, and
        /// `model download` on the server's machine still fetches. Off
        /// refuses POST /v1/models/remove too, the Server view's Remove.
        from_ui: Bool = true, NO_RULE, ALL, [Visible];
        /// What a download must leave free on the volume of `path` beyond
        /// the bytes it still misses: the Server view's Download, `model
        /// download` and the start's own fetch are refused, before any byte
        /// is written, when the free space is below the two together.
        free_margin: Bytes = super::DEFAULT_DOWNLOAD_FREE_MARGIN, NO_RULE, ALL, [Visible];
    }
}

config_group! {
    /// KV-disk, the tier below KV-RAM: evicted sequences and retained prompt
    /// checkpoints kept as files and read back instead of prefilled again.
    KvDiskGroup = "kv_disk" {
        /// KV-disk budget, capped at startup to the volume's free space minus 10 GiB.
        /// Defaults to 0 (disabled).
        bytes: Opt<Bytes> = None, NO_RULE, ALL, [Visible, Patchable, ReloadRequired, Scoped];
        /// Where KV-disk's files go: `model` (beside the artifact), `auto`
        /// (the per-user cache directory under kv-disk), or a directory. They
        /// go in an ignis-kv-disk/<pid>-<nonce> directory there, removed at
        /// shutdown.
        path: Location = CacheLocation::Model, NO_RULE, ALL, [Visible, Patchable, ReloadRequired];
    }
}

settings! {
    /// The HTTP server.
    server: ServerGroup,
    /// The model and its load shape.
    model: ModelGroup,
    /// Device memory.
    vram: VramGroup,
    /// Prompt reuse and KV-RAM.
    reuse: ReuseGroup,
    /// The model switch.
    switch: SwitchGroup,
    /// Speculative decoding.
    spec: SpecGroup,
    /// The vision tower.
    vision: VisionGroup,
    /// Media acquisition.
    media: MediaGroup,
    /// Model download.
    download: DownloadGroup,
    /// The n-gram table.
    /// KV-disk.
    kv_disk: KvDiskGroup,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The struct's real field names, read from its derived `Debug` — which
    /// rustc writes from the struct definition, not from the macro's table —
    /// so this compares two independent lists.
    fn debug_field_names(value: &impl std::fmt::Debug) -> Vec<String> {
        format!("{value:#?}")
            .lines()
            .filter_map(|line| {
                let rest = line.strip_prefix("    ")?;
                if rest.starts_with(' ') {
                    return None;
                }
                let (name, _) = rest.split_once(':')?;
                name.chars().all(|c| c.is_ascii_lowercase() || c == '_').then(|| name.to_owned())
            })
            .collect()
    }

    /// Spec config-v2/01, Testing: every group's table has exactly one entry
    /// per struct field and no orphan — the drift `IGNIS_PROMETHEUS` was.
    #[test]
    fn every_group_table_has_exactly_one_entry_per_struct_field() {
        let groups: Vec<(&str, Vec<String>)> = vec![
            (ServerGroup::NAME, debug_field_names(&ServerGroup::default())),
            (ModelGroup::NAME, debug_field_names(&ModelGroup::default())),
            (VramGroup::NAME, debug_field_names(&VramGroup::default())),
            (ReuseGroup::NAME, debug_field_names(&ReuseGroup::default())),
            (SwitchGroup::NAME, debug_field_names(&SwitchGroup::default())),
            (SpecGroup::NAME, debug_field_names(&SpecGroup::default())),
            (VisionGroup::NAME, debug_field_names(&VisionGroup::default())),
            (MediaGroup::NAME, debug_field_names(&MediaGroup::default())),
            (DownloadGroup::NAME, debug_field_names(&DownloadGroup::default())),
            (KvDiskGroup::NAME, debug_field_names(&KvDiskGroup::default())),
        ];
        assert_eq!(groups.len(), GROUPS.len(), "a group missing from one of the two lists");
        for ((name, struct_fields), (table_name, table)) in groups.iter().zip(GROUPS) {
            assert_eq!(name, table_name);
            let table_fields: Vec<String> = table.iter().map(|meta| meta.name.to_owned()).collect();
            assert!(!struct_fields.is_empty(), "{name}: no field read from Debug");
            assert_eq!(&table_fields, struct_fields, "{name}");
        }
        assert_eq!(
            debug_field_names(&Settings::default()),
            GROUPS.iter().map(|(name, _)| name.to_string()).collect::<Vec<_>>(),
            "every group in Settings, in order"
        );
    }

    /// The eleven groups spec config-v2/01 §Groups names, the image
    /// model's (spec image-model/05), the speech model's (spec speech/02) and
    /// the diarizer's (spec speech/08), no more.
    #[test]
    fn the_groups_are_the_specs() {
        let names: Vec<&str> = GROUPS.iter().map(|(name, _)| *name).collect();
        assert_eq!(
            names,
            ["server", "model", "vram", "reuse", "switch", "spec", "vision", "media", "download", "kv_disk"]
        );
    }

    /// No two fields share a spelling on any surface, family-scoped ones
    /// included, and none takes a name the CLI keeps for itself.
    #[test]
    fn every_flag_env_var_and_file_key_is_unique() {
        let mut flags = BTreeSet::new();
        let mut envs = BTreeSet::new();
        let mut keys = BTreeSet::new();
        for meta in all_fields() {
            assert!(flags.insert(meta.flag()), "{}", meta.flag());
            assert!(envs.insert(meta.env()), "{}", meta.env());
            assert!(keys.insert(meta.file_key()), "{}", meta.file_key());
            if meta.scoped {
                for family in super::super::field::FAMILIES {
                    assert!(flags.insert(meta.scoped_flag(family)), "{}", meta.scoped_flag(family));
                    assert!(envs.insert(meta.scoped_env(family)), "{}", meta.scoped_env(family));
                    assert!(keys.insert(meta.scoped_file_key(family)), "{}", meta.scoped_file_key(family));
                }
            }
        }
        for reserved in ["--config", "--profile", "--help", "--version"] {
            assert!(!flags.contains(reserved), "{reserved}");
        }
        for reserved in [super::super::CONFIG_ENV, super::super::PROFILE_ENV] {
            assert!(!envs.contains(reserved), "{reserved}");
        }
    }

    /// Spec config-v2/02 AC 2 and 6: the key is never visible nor patchable;
    /// the bind address is visible and not patchable; nothing is
    /// `ReloadRequired` without being patchable (the attribute would mean
    /// nothing).
    #[test]
    fn the_secret_is_hidden_and_the_sockets_are_not_patchable() {
        let api_key = field("server", "api_key").unwrap();
        assert!(!api_key.visible && !api_key.patchable);
        // Spec model-download/02 AC 7: the download token is hidden as the
        // key is, and no download field changes while the server runs.
        let token = field("download", "token").unwrap();
        assert!(!token.visible && !token.patchable);
        for name in ["enabled", "path", "endpoint", "catalog"] {
            let meta = field("download", name).unwrap();
            assert!(meta.visible && !meta.patchable, "{name}");
        }
        let bind = field("server", "bind").unwrap();
        assert!(bind.visible && !bind.patchable);
        for name in ["metrics_bind", "ui"] {
            assert!(!field("server", name).unwrap().patchable, "{name}");
        }
        for meta in all_fields() {
            assert!(!meta.reload_required || meta.patchable, "{}", meta.file_key());
        }
    }

    /// A field declared with no attributes at all is closed on every side:
    /// spec config-v2/02's fail-closed default, pinned on a throwaway
    /// declaration so no real field has to be made hidden to prove it.
    #[test]
    #[allow(dead_code)]
    fn a_field_given_no_attribute_is_hidden_unpatchable_and_unscoped() {
        config_group! {
            /// A throwaway group.
            Throwaway = "throwaway" {
                /// Declared with no attribute list.
                plain: Bool = false, NO_RULE, ALL;
                /// Declared visible only.
                shown: Bool = false, NO_RULE, ALL, [Visible];
            }
        }
        let plain = &Throwaway::FIELDS[0];
        assert!(!plain.visible && !plain.patchable && !plain.reload_required && !plain.scoped);
        assert!(Throwaway::FIELDS[1].visible && !Throwaway::FIELDS[1].patchable);
        assert_eq!(plain.summary(), "Declared with no attribute list.");
    }

    /// Spec config-v2/01, Testing: a pinned regression per field with a real
    /// validator today.
    #[test]
    fn the_validators_that_existed_before_are_declared() {
        assert_eq!(field("model", "prefill_chunk").unwrap().validator, Validator::MultipleOf(128));
        assert_eq!(field("spec", "decode_lanes").unwrap().validator, between(1, 8));
        assert!(matches!(field("model", "kv_format").unwrap().validator, Validator::OneOf(names) if names.contains(&"bf16") && names.contains(&"hq-e8-2b")));
        assert_eq!(field("server", "request_timeout").unwrap().validator, between(0, 3600));
    }

    /// The `OneOf` lists are literal (a `const` cannot call `as_str`); each
    /// is checked here against the enum it spells, so a new variant fails a
    /// test instead of being refused by its own validator.
    #[test]
    fn every_one_of_list_names_exactly_what_its_kind_parses() {
        let names = |group, name| match field(group, name).unwrap().validator {
            Validator::OneOf(names) => names.to_vec(),
            other => panic!("{group}.{name}: {other:?}"),
        };
        assert_eq!(names("server", "system_message_policy"), SystemMessagePolicy::ALL.map(|p| p.as_str()).to_vec());
        assert_eq!(names("server", "developer_message_policy"), DeveloperMessagePolicy::ALL.map(|p| p.as_str()).to_vec());
        assert_eq!(
            names("model", "reasoning_effort"),
            crate::thinking::ReasoningEffort::ALL.map(|e| e.as_str()).to_vec()
        );
        assert_eq!(names("server", "timeout_reply"), crate::deadline::TimeoutReply::ALL.map(|r| r.as_str()).to_vec());
        for name in names("model", "kv_format") {
            assert!(KvFormat::parse(name).is_ok(), "{name}");
        }
        assert_eq!(names("model", "prefill_attention"), PrefillAttention::NAMES.to_vec());
        assert_eq!(names("model", "exception_weights"), ExceptionWeights::NAMES.to_vec());
        for name in names("spec", "backend") {
            assert!(SpecBackend::parse(name).is_ok(), "{name}");
        }
        for name in names("spec", "draft_head") {
            assert!(DraftHead::parse(name).is_ok(), "{name}");
        }
    }

    /// `help --fields` prints every field's description: none may be empty.
    #[test]
    fn every_field_has_a_description() {
        for meta in all_fields() {
            assert!(!meta.summary().is_empty(), "{}", meta.file_key());
        }
    }
}
