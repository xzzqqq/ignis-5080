//! `config::resolve` end to end: every field by flag and by env var, the
//! rules that tie fields together, and the fit to a model family. The
//! behaviour each test pins is the one the flat flag set had before ADR 0046
//! regrouped it; only the spellings moved.

use super::*;
use crate::instruction::{DeveloperMessagePolicy, SystemMessagePolicy};
use crate::thinking::ReasoningEffort;
use ignis_core::compute::ModelFamily;
use ignis_core::speculation::{
    FLASH_NEXT_DEFAULT_DRAFT_TOKENS, FLASH_NEXT_DEFAULT_VERIFY_ROWS, FLASH_NEXT_MAX_VERIFY_ROWS,
};

fn no_env(_: &str) -> Option<String> {
    None
}

fn env_map(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
    move |key| pairs.iter().find(|(k, _)| *k == key).map(|(_, v)| v.to_string())
}

fn args(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

fn expect_config(outcome: ConfigOutcome) -> Config {
    match outcome {
        ConfigOutcome::Config(c) => c,
        other => panic!("expected Config, got {other:?}"),
    }
}

fn config(flags: &[&str]) -> Config {
    expect_config(resolve(&args(flags), no_env).expect("resolve"))
}

fn refused(flags: &[&str]) -> String {
    resolve(&args(flags), no_env).expect_err("must be refused").0
}

fn help() -> String {
    let ConfigOutcome::Help(text) = resolve(&args(&["--help"]), no_env).expect("resolve") else {
        panic!("--help is help");
    };
    text
}

/// A field's two help lines (flag, env and default; then its summary).
fn help_block(flag: &str) -> String {
    let text = help();
    let lines: Vec<&str> = text.lines().collect();
    let at = lines
        .iter()
        .position(|line| line.trim_start().starts_with(&format!("{flag} ")))
        .unwrap_or_else(|| panic!("help documents {flag}:\n{text}"));
    format!("{}\n{}", lines[at], lines.get(at + 1).unwrap_or(&""))
}

// ── the config file (spec config-v2/01 §The config file) ─────────────────

use file::testing::MemFiles;
use file::Format;

fn with_files(flags: &[&str], env: impl Fn(&str) -> Option<String>, files: &MemFiles) -> Result<Config, ConfigError> {
    resolve_with(&args(flags), env, files).map(expect_config)
}

/// Spec config-v2/01, Testing: a file written with every default reads back
/// to the same config in both formats — and starts either model, though it
/// names fields only one of them has (at their defaults).
#[test]
fn a_file_of_every_default_reads_back_to_the_defaults_in_both_formats() {
    let defaults = config(&["--profile", "none"]);
    let mut document = defaults.settings().render();
    document.insert("profile".to_owned(), serde_json::Value::String("none".into()));
    let document = serde_json::Value::Object(document);
    for (name, format) in [("c.yaml", Format::Yaml), ("c.json", Format::Json)] {
        let files = MemFiles::with(&[(name, &format.write(&document))]);
        let read = with_files(&["--config", name], no_env, &files).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(read.settings(), defaults.settings(), "{name}");
        assert_eq!(read.basis.sources().file_source, source::FileSource::Explicit(PathBuf::from(name)));
        for family in field::FAMILIES {
            read.for_family(family).unwrap_or_else(|e| panic!("{name} on {}: {e}", family.name()));
        }
    }
}

#[test]
fn a_file_value_sits_below_the_environment_and_the_flags() {
    let files = MemFiles::with(&[(
        "c.yaml",
        "reuse:\n  kv_host_pool_bytes: 3G\nmodel:\n  max_context: 65536\n  prefill_chunk: 512\n",
    )]);
    let env = env_map(&[("IGNIS_MODEL_MAX_CONTEXT", "32768")]);
    let config = with_files(&["--config", "c.yaml", "--model-prefill-chunk", "256"], env, &files).unwrap();
    assert_eq!(config.host_pool_bytes, 3 * GIB, "the file over the default");
    assert_eq!(config.max_context, 32_768, "the env over the file");
    assert_eq!(config.prefill_chunk, 256, "the flag over the file");
}

#[test]
fn a_file_is_named_by_the_flag_or_the_env_var_and_must_exist() {
    let files = MemFiles::with(&[("e.json", r#"{"server": {"request_timeout": 77}}"#)]);
    let env = env_map(&[("IGNIS_CONFIG", "e.json")]);
    assert_eq!(with_files(&[], env, &files).unwrap().request_timeout_secs, 77);
    let err = with_files(&["--config", "missing.yaml"], no_env, &files).unwrap_err().0;
    assert!(err.contains("missing.yaml"), "{err}");
    let err = with_files(&[], env_map(&[("IGNIS_CONFIG", "gone.yaml")]), &files).unwrap_err().0;
    assert!(err.contains("gone.yaml"), "an operator who named a file meant it to exist: {err}");
    assert!(resolve(&args(&["--config", "e.json"]), no_env).is_err(), "`resolve` reads no filesystem");
}

/// Spec config-v2/01, Testing: an unknown field or the wrong type is a
/// `ConfigError` naming the field.
#[test]
fn a_file_naming_an_unknown_field_or_a_bad_value_is_refused_by_name() {
    for (text, says) in [
        ("server:\n  bnd: x\n", "server.bnd"),
        ("model:\n  max_context: lots\n", "model.max_context"),
        ("model:\n  max_context: [1, 2]\n", "model.max_context"),
        ("vram:\n  headroom_bytes: true\n", "vram.headroom_bytes"),
        ("model: [\n", "not valid"),
    ] {
        let files = MemFiles::with(&[("c.yaml", text)]);
        let err = with_files(&["--config", "c.yaml"], no_env, &files).unwrap_err().0;
        assert!(err.contains(says) && err.contains("c.yaml"), "{text}: {err}");
    }
}

/// Every group struct and `Settings` derive their serde through the field
/// table: JSON and YAML round-trip, and a bad document is refused by name.
#[test]
fn the_settings_and_every_group_serialize_and_deserialize_in_both_formats() {
    let settings = config(&["--reuse-kv-host-pool-bytes", "3G", "--switch-known-models", "a=F:/a.ninfer", "--server-api-key", "k"])
        .settings()
        .clone();
    let json = serde_json::to_string(&settings).unwrap();
    assert_eq!(serde_json::from_str::<schema::Settings>(&json).unwrap(), settings);
    let yaml = serde_yaml::to_string(&settings).unwrap();
    assert_eq!(serde_yaml::from_str::<schema::Settings>(&yaml).unwrap(), settings);
    let reuse: schema::ReuseGroup = serde_json::from_str(r#"{"kv_host_pool_bytes": "3G"}"#).unwrap();
    assert_eq!(reuse, settings.reuse);
    let err = serde_json::from_str::<schema::ReuseGroup>(r#"{"kv_host_pool": 1}"#).unwrap_err().to_string();
    assert!(err.contains("reuse.kv_host_pool"), "{err}");
}

// ── config-file auto-discovery (spec config-v2/02) ───────────────────────

/// The per-user config directory's candidate, as discovery spells it here.
fn user_candidate() -> (&'static [(&'static str, &'static str)], &'static str) {
    if cfg!(windows) {
        (&[("APPDATA", "U")], "U/ignis/config.yaml")
    } else {
        (&[("XDG_CONFIG_HOME", "U")], "U/ignis/config.yaml")
    }
}

/// Spec config-v2/02, Testing: the working directory is found before the
/// user config directory when both hold a file; either is used when it is
/// the only one; none found is the no-file case, not an error.
#[test]
fn discovery_finds_the_working_directory_first_then_the_user_directory() {
    let (env, user_path) = user_candidate();
    let user = PathBuf::from("U").join("ignis").join("config.yaml");
    assert_eq!(user, PathBuf::from(user_path).components().collect::<PathBuf>());
    let both = MemFiles::with(&[("ignis.config.yaml", "server:\n  request_timeout: 11\n")]);
    both.files.lock().unwrap().insert(user.clone(), "server:\n  request_timeout: 22\n".to_owned());
    let found = with_files(&[], env_map(env), &both).unwrap();
    assert_eq!(found.request_timeout_secs, 11);
    assert_eq!(found.basis.sources().file_source, source::FileSource::Discovered(PathBuf::from("ignis.config.yaml")));

    let json_only = MemFiles::with(&[("ignis.config.json", r#"{"server": {"request_timeout": 33}}"#)]);
    assert_eq!(with_files(&[], env_map(env), &json_only).unwrap().request_timeout_secs, 33);

    let user_only = MemFiles::default();
    user_only.files.lock().unwrap().insert(user.clone(), "server:\n  request_timeout: 22\n".to_owned());
    let found = with_files(&[], env_map(env), &user_only).unwrap();
    assert_eq!((found.request_timeout_secs, found.basis.sources().file_source.kind()), (22, "discovered"));

    let none = with_files(&[], env_map(env), &MemFiles::default()).unwrap();
    assert_eq!(none.basis.sources().file_source, source::FileSource::None);
}

/// Spec config-v2/02, Testing: a file named by `--config` or `IGNIS_CONFIG`
/// short-circuits discovery entirely — no candidate is even looked at.
#[test]
fn a_named_file_short_circuits_discovery_without_looking() {
    let (env, _) = user_candidate();
    let files = MemFiles::with(&[("named.yaml", "server: {}\n"), ("ignis.config.yaml", "server:\n  request_timeout: 11\n")]);
    let named = with_files(&["--config", "named.yaml"], env_map(env), &files).unwrap();
    assert_eq!(named.request_timeout_secs, DEFAULT_REQUEST_TIMEOUT_SECS);
    assert_eq!(named.basis.sources().file_source.kind(), "explicit");
    assert!(files.looked_at.lock().unwrap().is_empty(), "{:?}", files.looked_at.lock().unwrap());
    let from_env = with_files(&[], env_map(&[("IGNIS_CONFIG", "named.yaml")]), &files).unwrap();
    assert_eq!(from_env.basis.sources().file_source, source::FileSource::Explicit(PathBuf::from("named.yaml")));
    assert!(files.looked_at.lock().unwrap().is_empty());
}

/// Spec config-v2/02, Testing: the startup line, for all three sources.
#[test]
fn the_startup_log_names_the_config_source_in_all_three_states() {
    use tracing_subscriber::layer::SubscriberExt;
    let (env, _) = user_candidate();
    let files = MemFiles::with(&[("named.yaml", "server: {}\n"), ("ignis.config.yaml", "server: {}\n")]);
    let cases = [
        (with_files(&["--config", "named.yaml"], env_map(env), &files).unwrap(), "explicit", "named.yaml"),
        (with_files(&[], env_map(env), &files).unwrap(), "discovered", "ignis.config.yaml"),
        (config(&[]), "none", "none"),
    ];
    for (config, source, path) in cases {
        let sink = std::sync::Arc::new(ignis_logging::MemorySink::new());
        let subscriber = tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone()));
        tracing::subscriber::with_default(subscriber, || log_source(&config));
        let lines = sink.lines();
        assert_eq!(lines.len(), 1, "{lines:?}");
        let event: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!(event["event_name"], "ignis.config.source", "{event}");
        let field = |name: &str| event.get(name).or_else(|| event["attributes"].get(name)).cloned();
        assert_eq!(field("source"), Some(source.into()), "{event}");
        assert_eq!(field("path"), Some(path.into()), "{event}");
        assert_eq!(field("profile"), Some("rtx50-16gb".into()), "{event}");
        assert_eq!(field("profile_by"), Some("auto".into()), "{event}");
        assert_eq!(field("card_total_bytes"), Some(profile::RTX_5080_TOTAL_BYTES.into()), "{event}");
    }
}

// ── hardware profiles (spec config-v2/02 §`--profile`) ───────────────────

/// The implicit default profile restates the hardcoded defaults: an unset
/// `--profile` changes nothing on the owner's machine.
#[test]
fn the_default_profile_changes_nothing() {
    let unset = config(&[]);
    let named = config(&["--profile", "rtx50-16gb"]);
    assert_eq!(unset.settings(), named.settings());
    assert_eq!(unset.basis.sources().profile_name, "rtx50-16gb");
    let fitted = unset.for_family(ModelFamily::Qwen38_27b).unwrap();
    let headroom = fitted.basis.resolution().origin("vram", "headroom_bytes").expect("the profile gave it");
    assert_eq!(headroom.source, source::Source::Profile);
}

#[test]
fn the_file_value_wins_over_its_own_profile_and_the_profile_is_named_by_flag_env_or_file() {
    let files = MemFiles::with(&[(
        "c.yaml",
        "profile: from-file\nvram:\n  headroom_bytes: 5G\nreuse:\n  kv_host_pool_bytes: 3G\nprofiles:\n  from-file:\n    reuse:\n      kv_host_pool_bytes: 1G\n      retained_device: 2\n  from-env:\n    reuse:\n      retained_device: 4\n",
    )]);
    let from_file = with_files(&["--config", "c.yaml"], no_env, &files).unwrap();
    assert_eq!(from_file.basis.sources().profile_name, "from-file");
    assert_eq!(from_file.host_pool_bytes, 3 * GIB, "the file over its profile");
    assert_eq!(from_file.retained_device_slots, 2);
    let env = env_map(&[("IGNIS_PROFILE", "from-env")]);
    assert_eq!(with_files(&["--config", "c.yaml"], env, &files).unwrap().retained_device_slots, 4, "the env var over the file's choice");
    let env = env_map(&[("IGNIS_PROFILE", "from-env")]);
    assert_eq!(with_files(&["--config", "c.yaml", "--profile", "from-file"], env, &files).unwrap().retained_device_slots, 2, "the flag over the env var");
    let err = with_files(&["--config", "c.yaml", "--profile", "h100"], no_env, &files).unwrap_err().0;
    assert!(err.contains("`h100`") && err.contains("from-env, from-file, rtx50-16gb"), "{err}");
    assert!(refused(&["--profile", "nope"]).contains("rtx50-16gb"));
}

/// KV-disk (spec vram-budget/03): unnamed is the family's budget and beside
/// the model; a size or `0`, `model`, `auto` or a directory by flag or
/// environment, the flag winning; a bad size is refused.
#[test]
fn kv_disk_defaults_flags_env_and_invalid_values() {
    use ignis_core::ngram_cache::CacheLocation;
    let default = config(&[]);
    assert_eq!(default.kv_disk_bytes, None, "the family's");
    assert_eq!(default.kv_disk_location, CacheLocation::Model);

    let env = env_map(&[("IGNIS_KV_DISK_BYTES", "8G"), ("IGNIS_KV_DISK_PATH", "auto")]);
    let from_env = expect_config(resolve(&[], &env).unwrap());
    assert_eq!(from_env.kv_disk_bytes, Some(8 << 30));
    assert_eq!(from_env.kv_disk_location, CacheLocation::Auto);

    let flags = args(&["--kv-disk-bytes", "0", "--kv-disk-path", "D:/kv"]);
    let from_flags = expect_config(resolve(&flags, &env).unwrap());
    assert_eq!(from_flags.kv_disk_bytes, Some(0), "0 is off, and the flag wins");
    assert_eq!(from_flags.kv_disk_location, CacheLocation::Directory(PathBuf::from("D:/kv")));

    let model = expect_config(resolve(&args(&["--kv-disk-path", "model"]), &env).unwrap());
    assert_eq!(model.kv_disk_location, CacheLocation::Model);

    for flags in [vec!["--kv-disk-bytes", "lots"], vec!["--kv-disk-bytes"], vec!["--kv-disk-path"]] {
        assert!(resolve(&args(&flags), no_env).is_err(), "{flags:?}");
    }
    assert!(resolve(&[], &env_map(&[("IGNIS_KV_DISK_BYTES", "-1")])).is_err());
    assert!(help().contains("--kv-disk-bytes") && help().contains("--kv-disk-path"));
}

#[test]
fn env_only_wins_over_defaults() {
    let env = env_map(&[
        ("IGNIS_MODEL_ID", "custom-model"),
        ("IGNIS_SERVER_BIND", "0.0.0.0:9000"),
        ("IGNIS_MODEL_ARTIFACT", "/path/to.ninfer"),
        ("IGNIS_MODEL_ENABLE_THINKING", "false"),
        ("IGNIS_MODEL_REASONING_EFFORT", "low"),
    ]);
    let config = expect_config(resolve(&[], env).expect("resolve"));
    assert_eq!(config.model, "custom-model");
    assert!(config.model_named);
    assert_eq!(config.bind, "0.0.0.0:9000");
    assert_eq!(config.artifact, Some(PathBuf::from("/path/to.ninfer")));
    assert!(!config.enable_thinking);
    assert_eq!(config.reasoning_effort, Some(ReasoningEffort::Low));
}

#[test]
fn flag_only_wins_over_defaults() {
    let config = config(&[
        "--model-id",
        "flag-model",
        "--server-bind",
        "0.0.0.0:1234",
        "--model-artifact",
        "/flag/artifact.ninfer",
        "--model-enable-thinking",
        "false",
        "--model-reasoning-effort",
        "high",
    ]);
    assert_eq!(config.model, "flag-model");
    assert_eq!(config.bind, "0.0.0.0:1234");
    assert_eq!(config.artifact, Some(PathBuf::from("/flag/artifact.ninfer")));
    assert!(!config.enable_thinking);
    assert_eq!(config.reasoning_effort, Some(ReasoningEffort::High));
}

/// ADR 0046 renamed every flag into its group, with one spelling per field:
/// the old short aliases went with the old names.
#[test]
fn the_old_short_aliases_and_flat_names_are_refused() {
    for flag in ["-m", "-b", "-a", "--model", "--bind", "--artifact", "--kv-host-pool-bytes", "--no-ui"] {
        let err = refused(&[flag, "x"]);
        assert!(err.contains(&format!("`{flag}`")), "{err}");
    }
    assert!(refused(&["--kv-host-pool-bytes", "8G"]).contains("`--reuse-kv-host-pool-bytes`"), "a hint at the new name");
}

#[test]
fn the_retired_telemetry_sink_flag_is_refused_and_its_env_var_ignored() {
    // ADR 0025: the interval counters are a log event now, so there is no
    // separate sink to point anywhere. A leftover `--telemetry` in a launch
    // script must fail loudly rather than be silently dropped.
    for flag in ["--telemetry", "-t"] {
        let err = refused(&[flag, "/tmp/telemetry.jsonl"]);
        assert!(err.contains(flag), "{err}");
    }
    let env = env_map(&[("IGNIS_TELEMETRY", "/tmp/telemetry.jsonl")]);
    assert_eq!(
        resolve(&[], env).expect("resolve"),
        resolve(&[], no_env).expect("resolve"),
        "IGNIS_TELEMETRY no longer changes the resolved config"
    );
    assert!(!help().contains("telemetry"), "help must not document it");
}

#[test]
fn a_flag_wins_over_a_matching_env_var_per_field_independently() {
    let env = env_map(&[("IGNIS_SERVER_BIND", "0.0.0.0:9000"), ("IGNIS_MODEL_ARTIFACT", "/env/artifact.ninfer")]);
    let config = expect_config(resolve(&args(&["--server-bind", "0.0.0.0:1234"]), env).expect("resolve"));
    assert_eq!(config.bind, "0.0.0.0:1234", "flag must win over env");
    assert_eq!(
        config.artifact,
        Some(PathBuf::from("/env/artifact.ninfer")),
        "env must still apply to a field the flag didn't touch"
    );
}

#[test]
fn an_unrecognized_flag_is_a_config_error() {
    assert!(refused(&["--nope"]).contains("--nope"));
}

#[test]
fn a_flag_missing_its_value_is_a_config_error() {
    assert!(refused(&["--server-bind"]).contains("--server-bind"));
}

#[test]
fn an_invalid_enable_thinking_is_refused_naming_where_it_came_from() {
    let flag_err = refused(&["--model-enable-thinking", "nope"]);
    assert!(flag_err.starts_with("`--model-enable-thinking` ") && flag_err.contains("`nope`"), "{flag_err}");
    let env_err = resolve(&[], env_map(&[("IGNIS_MODEL_ENABLE_THINKING", "nope")])).unwrap_err().0;
    assert_eq!(
        env_err.replacen("IGNIS_MODEL_ENABLE_THINKING", "--model-enable-thinking", 1),
        flag_err,
        "one reason, whichever way it arrived"
    );
}

#[test]
fn the_thinking_budget_ships_on_and_off_turns_it_off() {
    // Spec server/08: a measured default, so that a client that knows
    // nothing of the extension still gets an answer at `xhigh`.
    assert_eq!(config(&[]).thinking_budget, Some(DEFAULT_THINKING_BUDGET));
    assert_eq!(DEFAULT_THINKING_BUDGET, 32_768);
    // An empty env var is an unset one, as for the other defaults.
    let from_empty = expect_config(resolve(&[], env_map(&[("IGNIS_MODEL_THINKING_BUDGET", "")])).expect("resolve"));
    assert_eq!(from_empty.thinking_budget, Some(DEFAULT_THINKING_BUDGET));
    // `off`, by flag or by env: no default budget at all.
    assert_eq!(config(&["--model-thinking-budget", "off"]).thinking_budget, None);
    let off = expect_config(resolve(&[], env_map(&[("IGNIS_MODEL_THINKING_BUDGET", "off")])).expect("resolve"));
    assert_eq!(off.thinking_budget, None);
    // A number, and the flag over the env var.
    let env = env_map(&[("IGNIS_MODEL_THINKING_BUDGET", "off")]);
    let flag = expect_config(resolve(&args(&["--model-thinking-budget", "12288"]), env).expect("resolve"));
    assert_eq!(flag.thinking_budget, Some(12288));
    let six = expect_config(resolve(&[], env_map(&[("IGNIS_MODEL_THINKING_BUDGET", "6144")])).expect("resolve"));
    assert_eq!(six.thinking_budget, Some(6144));
}

#[test]
fn a_thinking_budget_that_is_neither_a_count_nor_off_refuses_the_start() {
    for bad in ["0", "-1", "lots", "8k", "OFF"] {
        let err = refused(&["--model-thinking-budget", bad]);
        assert!(err.contains("off") && err.contains(bad), "{bad}: the message names the way to say none: {err}");
    }
}

#[test]
fn the_help_names_the_thinking_budget_default_and_off() {
    let block = help_block("--model-thinking-budget");
    assert!(block.contains("IGNIS_MODEL_THINKING_BUDGET"), "{block}");
    assert!(block.contains(&format!("default: {DEFAULT_THINKING_BUDGET}")), "{block}");
    assert!(block.contains("off"), "{block}");
}

#[test]
fn an_invalid_reasoning_effort_is_refused_naming_every_value() {
    let err = refused(&["--model-reasoning-effort", "nonsense"]);
    assert!(err.contains("--model-reasoning-effort") && err.contains("nonsense") && err.contains("xhigh"), "{err}");
    let env_err = resolve(&[], env_map(&[("IGNIS_MODEL_REASONING_EFFORT", "nonsense")])).unwrap_err().0;
    assert!(env_err.starts_with("`IGNIS_MODEL_REASONING_EFFORT`"), "{env_err}");
}

#[test]
fn help_short_circuits_before_other_flags_are_validated() {
    assert!(matches!(resolve(&args(&["--help", "--nonsense"]), no_env).expect("resolve"), ConfigOutcome::Help(_)));
}

#[test]
fn help_short_circuits_even_where_it_would_otherwise_be_consumed_as_a_value() {
    // `--server-bind` normally requires a following value; `--help` still
    // wins rather than being swallowed as that value.
    assert!(matches!(resolve(&args(&["--server-bind", "--help"]), no_env).expect("resolve"), ConfigOutcome::Help(_)));
}

#[test]
fn help_alias_short_circuits_too() {
    assert!(matches!(resolve(&args(&["-h"]), no_env).expect("resolve"), ConfigOutcome::Help(_)));
}

#[test]
fn version_short_circuits_before_other_flags_are_validated() {
    assert!(matches!(resolve(&args(&["--version", "--nonsense"]), no_env).expect("resolve"), ConfigOutcome::Version(_)));
}

#[test]
fn version_alias_short_circuits_too() {
    assert!(matches!(resolve(&args(&["-V"]), no_env).expect("resolve"), ConfigOutcome::Version(_)));
}

// ── the engine-shape fields (GitHub #87) ─────────────────────────────────

#[test]
fn the_default_context_admits_a_32k_prompt_plus_a_generation_budget() {
    // G2's largest cell is a 32,768-token prompt; the default cap must admit
    // it *and* leave room to generate, without editing code.
    let config = config(&[]);
    assert!(config.max_context > 32_768, "{}", config.max_context);
    // The pool is the policy's, which the load's plan refuses when it cannot
    // hold one such sequence (GitHub #210, ADR 0045).
    assert_eq!(config.kv_pool, None);
}

#[test]
fn the_engine_shape_env_vars_win_over_the_defaults() {
    let env = env_map(&[("IGNIS_MODEL_PREFILL_CHUNK", "2048"), ("IGNIS_MODEL_MAX_CONTEXT", "16384")]);
    let config = expect_config(resolve(&[], env).expect("resolve"));
    assert_eq!(config.prefill_chunk, 2048);
    assert_eq!(config.max_context, 16_384);
}

#[test]
fn the_engine_shape_flags_win_over_their_env_vars() {
    let env = env_map(&[("IGNIS_MODEL_PREFILL_CHUNK", "2048"), ("IGNIS_MODEL_MAX_CONTEXT", "16384")]);
    let a = args(&["--model-prefill-chunk", "128", "--model-max-context", "8192"]);
    let config = expect_config(resolve(&a, env).expect("resolve"));
    assert_eq!(config.prefill_chunk, 128, "flag must win over env");
    assert_eq!(config.max_context, 8_192, "flag must win over env");
}

#[test]
fn the_decode_share_is_a_percent_below_100_and_unset_by_default() {
    // GitHub #306: unset, the model family's own (`EngineShape::for_family`).
    assert_eq!(config(&[]).decode_share_percent, None);
    assert_eq!(config(&["--model-decode-share", "50"]).decode_share_percent, Some(50));
    let zero = expect_config(resolve(&[], env_map(&[("IGNIS_MODEL_DECODE_SHARE", "0")])).expect("resolve"));
    assert_eq!(zero.decode_share_percent, Some(0), "0 is ADR 0018's one round per chunk");
    for bad in ["100", "150", "-1", "0.5", "half"] {
        let err = refused(&["--model-decode-share", bad]);
        assert!(err.contains("--model-decode-share") && err.contains(bad), "{bad}: {err}");
    }
}

#[test]
fn an_unaligned_prefill_chunk_is_a_usage_error() {
    // The alignment rule is the reference's own; an unaligned width is
    // rejected before any loader work, not at the first long prompt.
    let err = refused(&["--model-prefill-chunk", "1000"]);
    assert!(err.contains("128"), "the message must name the rule: {err}");
    assert!(err.contains("1000"), "the message must name the value: {err}");
}

#[test]
fn a_zero_prefill_chunk_is_a_usage_error() {
    assert!(refused(&["--model-prefill-chunk", "0"]).contains("nonzero"));
}

#[test]
fn a_non_numeric_prefill_chunk_is_a_usage_error() {
    assert!(refused(&["--model-prefill-chunk", "wide"]).contains("--model-prefill-chunk"));
}

#[test]
fn an_invalid_prefill_chunk_env_var_is_a_usage_error_too() {
    // Same rule whichever way the value arrived: the env var is not a back
    // door around the validation.
    let err = resolve(&[], env_map(&[("IGNIS_MODEL_PREFILL_CHUNK", "300")])).expect_err("must reject");
    assert!(err.0.contains("128") && err.0.contains("IGNIS_MODEL_PREFILL_CHUNK"), "{err}");
}

#[test]
fn a_zero_max_context_is_a_usage_error() {
    assert!(refused(&["--model-max-context", "0"]).contains("--model-max-context"));
}

// ── the default max_tokens (ADR 0045, GitHub #309) ───────────────────────

#[test]
fn the_default_max_tokens_is_38912_and_takes_a_value_from_every_spelling() {
    assert_eq!(config(&[]).default_max_tokens, 38_912);
    // AC 10: a non-default value, 8,192, from the flag and from the env var
    // in turn (make's knob: `mk/flags-selftest.sh`).
    assert_eq!(config(&["--model-default-max-tokens", "8192"]).default_max_tokens, 8_192);
    let env = env_map(&[("IGNIS_MODEL_DEFAULT_MAX_TOKENS", "8192")]);
    assert_eq!(expect_config(resolve(&[], &env).expect("resolve")).default_max_tokens, 8_192);
    let flag = expect_config(resolve(&args(&["--model-default-max-tokens", "4096"]), &env).expect("resolve"));
    assert_eq!(flag.default_max_tokens, 4_096, "the flag wins over the env var");
    assert_eq!(config(&["--model-default-max-tokens", "0"]).default_max_tokens, 0, "0 is none");
    // Past the context it is accepted: the scheduler clamps it to what each
    // prompt leaves, so it acts as the context.
    let past = config(&["--model-max-context", "40960", "--model-default-max-tokens", "1000000"]);
    assert_eq!(past.default_max_tokens, 1_000_000);
}

#[test]
fn a_malformed_default_max_tokens_is_refused_naming_where_it_came_from() {
    for bad in ["eight", "-1", "8k", "1.5"] {
        assert!(refused(&["--model-default-max-tokens", bad]).contains("--model-default-max-tokens"), "{bad}");
    }
    let err = resolve(&[], env_map(&[("IGNIS_MODEL_DEFAULT_MAX_TOKENS", "lots")])).expect_err("env");
    assert!(err.0.contains("IGNIS_MODEL_DEFAULT_MAX_TOKENS"), "{err}");
}

#[test]
fn the_help_names_the_default_max_tokens_and_none() {
    let block = help_block("--model-default-max-tokens");
    assert!(block.contains("IGNIS_MODEL_DEFAULT_MAX_TOKENS"), "{block}");
    assert!(block.contains("default: 38912"), "{block}");
    assert!(block.contains("0 allows output up to the remaining context"), "{block}");
}

#[test]
fn an_unnamed_pool_is_left_to_the_vram_plan_at_any_context() {
    // No auto budget to outgrow any more (GitHub #210): the pool is what the
    // VRAM budget leaves, in either format and at any cap, and the plan —
    // not this config — refuses a start where that is short.
    for (format, context) in [("bf16", 200_000u32), ("hq-e8-2b", 600_000)] {
        let config = config(&["--model-kv-format", format, "--model-max-context", &context.to_string()]);
        assert_eq!(config.max_context, context);
        assert_eq!(config.kv_pool, None, "{format} at {context}");
    }
}

#[test]
fn help_lists_the_engine_shape_flags() {
    let text = help();
    for flag in ["--model-prefill-chunk", "--model-max-context", "--model-kv-format", "--vram-kv-pool-bytes"] {
        assert!(text.contains(flag), "help must document {flag}:\n{text}");
    }
    assert!(text.contains("hq-e8-2b"), "help must name both formats:\n{text}");
}

// ── the KV format and pool budget (GitHub #122) ──────────────────────────

#[test]
fn the_kv_format_flag_wins_over_the_env_var_and_the_default() {
    let env = env_map(&[("IGNIS_MODEL_KV_FORMAT", "bf16")]);
    let flag = expect_config(resolve(&args(&["--model-kv-format", "hq-e8-2b"]), env).expect("resolve"));
    assert_eq!(flag.kv_format, KvFormat::HqE8_2b);
    // Both halves name the format the default is *not*, so neither can pass
    // by agreeing with it (GitHub #123 made the default hq-e8-2b).
    let env = env_map(&[("IGNIS_MODEL_KV_FORMAT", "bf16")]);
    assert_eq!(expect_config(resolve(&[], env).expect("resolve")).kv_format, KvFormat::Bf16);
}

#[test]
fn an_unknown_kv_format_is_a_usage_error() {
    let err = refused(&["--model-kv-format", "fp8"]);
    assert!(err.contains("--model-kv-format") && err.contains("fp8") && err.contains("bf16"), "{err}");
}

// ── the prompt attention precision (GitHub #335) ─────────────────────────

#[test]
fn the_prefill_attention_is_fp8_unless_named_and_flag_beats_env() {
    assert_eq!(config(&[]).prefill_attention, PrefillAttention::Fp8);
    let env = env_map(&[("IGNIS_MODEL_PREFILL_ATTENTION", "fp16")]);
    assert_eq!(expect_config(resolve(&[], env).expect("resolve")).prefill_attention, PrefillAttention::Fp16);
    let env = env_map(&[("IGNIS_MODEL_PREFILL_ATTENTION", "fp16")]);
    let flag = expect_config(resolve(&args(&["--model-prefill-attention", "FP8"]), env).expect("resolve"));
    assert_eq!(flag.prefill_attention, PrefillAttention::Fp8);
}

#[test]
fn an_unknown_prefill_attention_is_a_usage_error() {
    let err = refused(&["--model-prefill-attention", "fp4"]);
    assert!(err.contains("--model-prefill-attention") && err.contains("fp4") && err.contains("fp16"), "{err}");
}

#[test]
fn the_same_named_budget_buys_more_tokens_under_hq() {
    // The format is a real option: one budget, two capacities. This is the
    // whole reason the pool is described in bytes.
    let geometry = ignis_core::KvGeometry::qwen38_27b();
    let named = |format| config(&["--model-kv-format", format, "--vram-kv-pool-bytes", "4G"]);
    let (bf16, hq) = (named("bf16"), named("hq-e8-2b"));
    assert_eq!(bf16.kv_pool, hq.kv_pool);
    let Some(KvPoolSize::Bytes(bytes)) = hq.kv_pool else {
        panic!("named in bytes: {:?}", hq.kv_pool);
    };
    let bf16_capacity = ignis_core::plan_kv_pool(bf16.kv_format, geometry, bytes).token_capacity;
    let hq_capacity = ignis_core::plan_kv_pool(hq.kv_format, geometry, bytes).token_capacity;
    assert!(hq_capacity > bf16_capacity * 7, "{hq_capacity} vs {bf16_capacity}");
    // And it clears the standard target profile: 8 lanes x 40,960.
    assert!(hq_capacity >= 8 * 40_960);
}

#[test]
fn a_named_pool_budget_resolves_from_the_flag_and_the_env() {
    assert_eq!(config(&["--vram-kv-pool-bytes", "8G"]).kv_pool, Some(KvPoolSize::Bytes(8 * 1024 * 1024 * 1024)));
    let env = env_map(&[("IGNIS_VRAM_KV_POOL_BYTES", "6144MiB")]);
    assert_eq!(expect_config(resolve(&[], env).expect("resolve")).kv_pool, Some(KvPoolSize::Bytes(6144 * 1024 * 1024)));
    // A bare count is still a byte count.
    assert_eq!(config(&["--vram-kv-pool-bytes", "4294967296"]).kv_pool, Some(KvPoolSize::Bytes(4 << 30)));
}

#[test]
fn a_named_pool_takes_a_token_count_with_binary_multipliers() {
    // ADR 0045 (AC 5): the quantity the owner decides in, the same context
    // on either model and either format.
    for (raw, tokens) in [("512Ktok", 524_288), ("2Mtok", 2 * 1024 * 1024), ("1000tok", 1_000), ("512ktok", 524_288), (" 64Ktok ", 65_536)] {
        assert_eq!(config(&["--vram-kv-pool-bytes", raw]).kv_pool, Some(KvPoolSize::Tokens(tokens)), "{raw}");
    }
    // The env var takes the same spellings, and the flag wins over it.
    let env = env_map(&[("IGNIS_VRAM_KV_POOL_BYTES", "512Ktok")]);
    assert_eq!(expect_config(resolve(&[], &env).expect("env")).kv_pool, Some(KvPoolSize::Tokens(524_288)));
    let flag = expect_config(resolve(&args(&["--vram-kv-pool-bytes", "4G"]), &env).expect("flag"));
    assert_eq!(flag.kv_pool, Some(KvPoolSize::Bytes(4 << 30)));
}

#[test]
fn the_config_only_parses_a_pool_and_the_plan_judges_it() {
    // ADR 0045: a byte count is a different context on each model, and only
    // the load knows which. The config accepts it in either format, and the
    // plan refuses it below one context and a page per retained slot
    // (`vram.rs`'s `a_named_pool_below_one_context_refuses_*`).
    for format in ["bf16", "hq-e8-2b"] {
        assert_eq!(config(&["--vram-kv-pool-bytes", "1M", "--model-kv-format", format]).kv_pool, Some(KvPoolSize::Bytes(1 << 20)));
    }
    // What the plan will make of 512 MiB on the 27B at the default context:
    // under hq it buys more than one context and the retained slots' pages,
    // under BF16 not -- the format still decides.
    let geometry = ignis_core::KvGeometry::qwen38_27b();
    let floor = DEFAULT_MAX_CONTEXT.div_ceil(64) + DEFAULT_RETAINED_HOST_SLOTS + DEFAULT_RETAINED_DEVICE_SLOTS;
    let pages = |format: KvFormat| KvPoolSize::Bytes(512 << 20).pages(format.page_bytes(geometry));
    assert!(pages(KvFormat::HqE8_2b) >= floor, "{} vs {floor}", pages(KvFormat::HqE8_2b));
    assert!(pages(KvFormat::Bf16) < floor, "{} vs {floor}", pages(KvFormat::Bf16));
}

#[test]
fn a_malformed_pool_budget_is_a_usage_error() {
    for raw in ["", "4 GiB please", "-1", "4TB", "G", "tok", "Ktok", "4Gtok", "1.5Ktok", "-1tok", "12 tok"] {
        match resolve(&args(&["--vram-kv-pool-bytes", raw]), no_env) {
            // An empty value is an unset one, the same as every other flag.
            Ok(config) if raw.is_empty() => assert_eq!(expect_config(config).kv_pool, None),
            Ok(_) => panic!("`{raw}` must not parse as a byte or token count"),
            Err(err) => assert!(err.0.contains("--vram-kv-pool-bytes"), "{}", err.0),
        }
    }
}

// ── the VRAM budget (GitHub #210, ADR 0030) ──────────────────────────────

const GIB: u64 = 1024 * 1024 * 1024;

#[test]
fn no_memory_flag_derives_the_budget_with_a_one_and_a_half_gib_headroom() {
    assert_eq!(config(&[]).vram, VramMode::Derived { headroom_bytes: 1536 * 1024 * 1024 });
}

#[test]
fn help_names_the_vram_headroom_default() {
    let block = help_block("--vram-headroom-bytes");
    assert!(block.contains("default: 1536M"), "{block}");
}

#[test]
fn the_headroom_and_the_budget_resolve_from_flags_and_env() {
    assert_eq!(config(&["--vram-headroom-bytes", "2G"]).vram, VramMode::Derived { headroom_bytes: 2 * GIB });
    let env = env_map(&[("IGNIS_VRAM_HEADROOM_BYTES", "512M")]);
    assert_eq!(expect_config(resolve(&[], env).expect("resolve")).vram, VramMode::Derived { headroom_bytes: 512 << 20 });
    assert_eq!(
        config(&["--vram-budget-bytes", "28G"]).vram,
        VramMode::Explicit { budget_bytes: 28 * GIB, allow_oversubscription: false }
    );
    let env = env_map(&[("IGNIS_VRAM_BUDGET_BYTES", "30G"), ("IGNIS_VRAM_ALLOW_OVERSUBSCRIPTION", "true")]);
    assert_eq!(
        expect_config(resolve(&[], env).expect("resolve")).vram,
        VramMode::Explicit { budget_bytes: 30 * GIB, allow_oversubscription: true }
    );
    assert_eq!(
        config(&["--vram-budget-bytes", "30G", "--vram-allow-oversubscription"]).vram,
        VramMode::Explicit { budget_bytes: 30 * GIB, allow_oversubscription: true }
    );
}

#[test]
fn a_headroom_with_a_budget_is_refused_from_any_mix_of_flags_and_env() {
    let cases: [(Vec<String>, &'static [(&'static str, &'static str)]); 4] = [
        (args(&["--vram-headroom-bytes", "1G", "--vram-budget-bytes", "28G"]), &[]),
        (args(&["--vram-budget-bytes", "28G"]), &[("IGNIS_VRAM_HEADROOM_BYTES", "1G")]),
        (args(&["--vram-headroom-bytes", "1G"]), &[("IGNIS_VRAM_BUDGET_BYTES", "28G")]),
        (vec![], &[("IGNIS_VRAM_HEADROOM_BYTES", "1G"), ("IGNIS_VRAM_BUDGET_BYTES", "28G")]),
    ];
    for (a, env) in cases {
        let err = resolve(&a, env_map(env)).expect_err("mutually exclusive").0;
        assert!(err.contains("HEADROOM") || err.contains("headroom"), "{err}");
        assert!(err.contains("BUDGET") || err.contains("budget"), "{err}");
        assert!(err.contains("mutually exclusive"), "{err}");
    }
}

#[test]
fn oversubscription_without_a_budget_is_refused() {
    let cases: [(Vec<String>, &'static [(&'static str, &'static str)]); 3] = [
        (args(&["--vram-allow-oversubscription"]), &[]),
        (vec![], &[("IGNIS_VRAM_ALLOW_OVERSUBSCRIPTION", "1")]),
        (args(&["--vram-allow-oversubscription", "--vram-headroom-bytes", "2G"]), &[]),
    ];
    for (a, env) in cases {
        let err = resolve(&a, env_map(env)).expect_err("needs a budget").0;
        assert!(err.contains("OVERSUBSCRIPTION") || err.contains("oversubscription"), "{err}");
        assert!(err.contains("--vram-budget-bytes"), "{err}");
    }
}

#[test]
fn malformed_vram_values_are_usage_errors() {
    for (flag, raw) in [("--vram-headroom-bytes", "lots"), ("--vram-budget-bytes", "28 GiB please"), ("--vram-budget-bytes", "0")] {
        assert!(refused(&[flag, raw]).contains(flag), "{flag} {raw}");
    }
    let env = env_map(&[("IGNIS_VRAM_BUDGET_BYTES", "28G"), ("IGNIS_VRAM_ALLOW_OVERSUBSCRIPTION", "maybe")]);
    assert!(resolve(&[], env).expect_err("not a bool").0.contains("IGNIS_VRAM_ALLOW_OVERSUBSCRIPTION"));
}

#[test]
fn help_documents_the_vram_flags() {
    let text = help();
    for name in [
        "--vram-headroom-bytes",
        "IGNIS_VRAM_HEADROOM_BYTES",
        "--vram-budget-bytes",
        "IGNIS_VRAM_BUDGET_BYTES",
        "--vram-allow-oversubscription",
        "IGNIS_VRAM_ALLOW_OVERSUBSCRIPTION",
    ] {
        assert!(text.contains(name), "help must document {name}:\n{text}");
    }
}

// ── the KV-RAM host tier byte budget (P4-07, GitHub #125) ────────────────

#[test]
fn an_explicit_host_pool_budget_overrides_the_default() {
    assert_eq!(config(&["--reuse-kv-host-pool-bytes", "512M"]).host_pool_bytes, 512 << 20);
    let env = env_map(&[("IGNIS_REUSE_KV_HOST_POOL_BYTES", "1G")]);
    assert_eq!(expect_config(resolve(&[], env).expect("resolve")).host_pool_bytes, GIB);
}

#[test]
fn the_host_pool_budget_flag_wins_over_its_env_var() {
    let env = env_map(&[("IGNIS_REUSE_KV_HOST_POOL_BYTES", "1G")]);
    let config = expect_config(resolve(&args(&["--reuse-kv-host-pool-bytes", "256M"]), env).expect("resolve"));
    assert_eq!(config.host_pool_bytes, 256 << 20, "flag must win over env");
}

#[test]
fn a_zero_host_pool_budget_is_accepted_and_disables_the_tier() {
    // Unlike an empty string (an unset value), `0` is an explicit, legal
    // operator choice: no host tier at all.
    assert_eq!(config(&["--reuse-kv-host-pool-bytes", "0"]).host_pool_bytes, 0);
}

// ── the host plan's margin (`--model-host-margin-bytes`) ──────────────────

#[test]
fn the_host_margin_is_four_gib_unless_named() {
    assert_eq!(DEFAULT_HOST_MARGIN_BYTES, 4 * GIB);
    assert_eq!(config(&["--profile", "none"]).host_margin_bytes, DEFAULT_HOST_MARGIN_BYTES);
    assert_eq!(config(&["--model-host-margin-bytes", "3G"]).host_margin_bytes, 3 * GIB);
    assert_eq!(config(&["--model-host-margin-bytes", "0"]).host_margin_bytes, 0, "no margin is the operator's call");
    let env = env_map(&[("IGNIS_MODEL_HOST_MARGIN_BYTES", "5G")]);
    assert_eq!(expect_config(resolve(&[], env).expect("resolve")).host_margin_bytes, 5 * GIB);
}

#[test]
fn a_malformed_host_pool_budget_is_a_usage_error() {
    assert!(refused(&["--reuse-kv-host-pool-bytes", "not-a-size"]).contains("--reuse-kv-host-pool-bytes"));
}

#[test]
fn help_lists_the_host_pool_budget_flag() {
    assert!(help().contains("--reuse-kv-host-pool-bytes"));
}

// ── GitHub #186: cross-request state reuse (ADR 0029) ────────────────────

/// (device, host) retained slots.
fn retained_slots_of(config: &Config) -> (u32, u32) {
    (config.retained_device_slots, config.retained_host_slots)
}

#[test]
fn prompt_reuse_is_on_unless_the_operator_turns_it_off() {
    let on = config(&[]);
    assert!(on.prompt_reuse, "on by default (ADR 0029)");
    assert_eq!(
        retained_slots_of(&on),
        (0, 2 * ignis_core::N_DECODE_LANES as u32),
        "no device slot and two host slots per decode lane by default (GitHub #281)"
    );
    let off = config(&["--reuse-prompt", "off"]);
    assert!(!off.prompt_reuse);
    assert_eq!(retained_slots_of(&off), (0, 0), "reuse off reserves no slot of either kind");
    let env = env_map(&[("IGNIS_REUSE_PROMPT", "off")]);
    assert!(!expect_config(resolve(&[], env).expect("resolve")).prompt_reuse, "the env var turns it off too");
    let env = env_map(&[("IGNIS_REUSE_PROMPT", "off")]);
    assert!(expect_config(resolve(&args(&["--reuse-prompt", "on"]), env).expect("resolve")).prompt_reuse, "flag must win over env");
}

#[test]
fn a_malformed_prompt_reuse_value_is_a_usage_error() {
    let err = refused(&["--reuse-prompt", "maybe"]);
    assert!(err.contains("--reuse-prompt") && err.contains("maybe"), "{err}");
}

#[test]
fn retained_device_and_host_slots_are_configurable_by_flag_or_env() {
    // GitHub #281: each kind is sized on its own, its default kept when only
    // the other is named.
    let host_default = 2 * ignis_core::N_DECODE_LANES as u32;
    assert_eq!(retained_slots_of(&config(&["--reuse-retained-device", "3"])), (3, host_default));
    assert_eq!(retained_slots_of(&config(&["--reuse-retained-host", "5"])), (0, 5));
    // A card with VRAM to spare keeps every image on the device.
    assert_eq!(retained_slots_of(&config(&["--reuse-retained-device", "16", "--reuse-retained-host", "0"])), (16, 0));
    let env = env_map(&[("IGNIS_REUSE_RETAINED_DEVICE", "12"), ("IGNIS_REUSE_RETAINED_HOST", "7")]);
    assert_eq!(retained_slots_of(&expect_config(resolve(&[], env).expect("resolve"))), (12, 7));
    let env = env_map(&[("IGNIS_REUSE_RETAINED_DEVICE", "12"), ("IGNIS_REUSE_RETAINED_HOST", "7")]);
    let a = args(&["--reuse-retained-device", "1", "--reuse-retained-host", "2"]);
    assert_eq!(retained_slots_of(&expect_config(resolve(&a, env).expect("resolve"))), (1, 2), "the flag wins over its env var");
    for flag in ["--reuse-retained-device", "--reuse-retained-host"] {
        assert!(refused(&[flag, "8G"]).contains(flag), "{flag}");
    }
}

#[test]
fn the_config_records_whether_the_host_retained_count_was_named() {
    // Spec flash-next/05: an unnamed count is the 27B's default here and
    // Flash-Next's own once the artifact names its model, so the config
    // keeps which of the two it is.
    assert!(!config(&[]).retained_host_named);
    assert!(!config(&["--reuse-retained-device", "2"]).retained_host_named, "the device count names nothing of the host's");
    assert!(config(&["--reuse-retained-host", "16"]).retained_host_named, "named at the 27B's default value is still named");
    let env = env_map(&[("IGNIS_REUSE_RETAINED_HOST", "3")]);
    assert!(expect_config(resolve(&[], env).expect("resolve")).retained_host_named, "the env var names it too");
}

#[test]
fn the_retained_interactive_ttl_is_configurable_and_needs_reuse_on() {
    assert_eq!(config(&[]).retained_interactive_ttl_secs, 300, "five minutes by default");
    assert_eq!(config(&["--reuse-retained-interactive-ttl", "60"]).retained_interactive_ttl_secs, 60);
    let env = env_map(&[("IGNIS_REUSE_RETAINED_INTERACTIVE_TTL", "0")]);
    assert_eq!(expect_config(resolve(&[], env).expect("resolve")).retained_interactive_ttl_secs, 0, "zero is a legal choice");
    assert!(refused(&["--reuse-retained-interactive-ttl", "5m"]).contains("--reuse-retained-interactive-ttl"));
    let err = refused(&["--reuse-prompt", "off", "--reuse-retained-interactive-ttl", "60"]);
    assert!(err.contains("--reuse-prompt"), "names what it needs: {err}");
}

#[test]
fn retained_slots_with_reuse_off_are_accepted_for_live_siblings() {
    // With reuse off nothing outlives a request, but live siblings still
    // share a head when slots are given for it — the engine before #186.
    let device = config(&["--reuse-prompt", "off", "--reuse-retained-device", "4"]);
    assert!(!device.prompt_reuse);
    assert_eq!(retained_slots_of(&device), (4, 0), "the kind not named stays at zero");
    assert_eq!(retained_slots_of(&config(&["--reuse-prompt", "off", "--reuse-retained-host", "3"])), (0, 3));
}

// ── GitHub #209: instruction-message policies ────────────────────────────

#[test]
fn instruction_policies_default_to_merge_and_inplace() {
    assert_eq!(
        config(&[]).instruction_policy,
        InstructionPolicy { system: SystemMessagePolicy::Merge, developer: DeveloperMessagePolicy::Inplace }
    );
}

#[test]
fn every_instruction_policy_value_is_named_by_flag_or_env_and_the_flag_wins() {
    for system in SystemMessagePolicy::ALL {
        assert_eq!(config(&["--server-system-message-policy", system.as_str()]).instruction_policy.system, system);
        let env = move |key: &str| (key == "IGNIS_SERVER_SYSTEM_MESSAGE_POLICY").then(|| system.as_str().to_owned());
        assert_eq!(expect_config(resolve(&[], env).expect("resolve")).instruction_policy.system, system);
    }
    for developer in DeveloperMessagePolicy::ALL {
        assert_eq!(config(&["--server-developer-message-policy", developer.as_str()]).instruction_policy.developer, developer);
        let env = move |key: &str| (key == "IGNIS_SERVER_DEVELOPER_MESSAGE_POLICY").then(|| developer.as_str().to_owned());
        assert_eq!(expect_config(resolve(&[], env).expect("resolve")).instruction_policy.developer, developer);
    }
    let env = env_map(&[("IGNIS_SERVER_SYSTEM_MESSAGE_POLICY", "strict"), ("IGNIS_SERVER_DEVELOPER_MESSAGE_POLICY", "reject")]);
    let a = args(&["--server-system-message-policy", "merge", "--server-developer-message-policy", "into-system"]);
    assert_eq!(
        expect_config(resolve(&a, env).expect("resolve")).instruction_policy,
        InstructionPolicy { system: SystemMessagePolicy::Merge, developer: DeveloperMessagePolicy::IntoSystem },
        "flags win over env"
    );
}

#[test]
fn an_unknown_instruction_policy_names_the_allowed_values() {
    let err = refused(&["--server-system-message-policy", "inplace"]);
    assert!(err.contains("--server-system-message-policy"), "{err}");
    assert!(err.contains("`inplace`"), "names the value: {err}");
    assert!(err.contains("merge, strict"), "names the allowed ones: {err}");

    let err = resolve(&[], env_map(&[("IGNIS_SERVER_DEVELOPER_MESSAGE_POLICY", "drop")])).expect_err("not a developer policy").0;
    assert!(err.contains("`drop`"), "names the value: {err}");
    assert!(err.contains("IGNIS_SERVER_DEVELOPER_MESSAGE_POLICY"), "names the env var it came from: {err}");
    assert!(err.contains("inplace, into-system, after-system, one-after-system, reject"), "names the allowed ones: {err}");

    let padded = config(&["--server-developer-message-policy", " One-After-System "]);
    assert_eq!(padded.instruction_policy.developer, DeveloperMessagePolicy::OneAfterSystem, "case and padding do not matter");
}

// ── the request timeout (GitHub #95) ─────────────────────────────────────

#[test]
fn the_request_timeout_env_var_wins_over_the_default() {
    let env = env_map(&[("IGNIS_SERVER_REQUEST_TIMEOUT", "90")]);
    assert_eq!(expect_config(resolve(&[], env).expect("resolve")).request_timeout_secs, 90);
}

#[test]
fn the_request_timeout_flag_wins_over_its_env_var() {
    let env = env_map(&[("IGNIS_SERVER_REQUEST_TIMEOUT", "90")]);
    let config = expect_config(resolve(&args(&["--server-request-timeout", "45"]), env).expect("resolve"));
    assert_eq!(config.request_timeout_secs, 45, "flag must win over env");
}

#[test]
fn a_request_timeout_outside_zero_to_the_ceiling_is_a_usage_error() {
    assert!(refused(&["--server-request-timeout", "soon"]).contains("--server-request-timeout"));
    let err = refused(&["--server-request-timeout", "3601"]);
    assert!(err.contains("0..=3600") && err.contains("3601"), "the range and the value: {err}");
    let err = resolve(&[], env_map(&[("IGNIS_SERVER_REQUEST_TIMEOUT", "3601")])).expect_err("must reject").0;
    assert!(err.contains("IGNIS_SERVER_REQUEST_TIMEOUT"), "{err}");
}

#[test]
fn help_lists_the_request_timeout_flag() {
    assert!(help().contains("--server-request-timeout"));
}

// ── request deadlines on every route (ADR 0058) ──────────────────────────

/// `server.request_timeout` (non-streaming) and `server.stream_timeout`
/// (streaming) both default to 40 minutes, take 0 for no deadline, stop at
/// the same ceiling, and are visible and patchable with no reload; the live
/// knobs carry them, and a server built without a config holds the same
/// defaults.
#[test]
fn both_deadlines_default_to_forty_minutes_take_zero_and_share_the_ceiling() {
    let defaults = config(&[]);
    assert_eq!((defaults.request_timeout_secs, defaults.stream_timeout_secs), (2400, 2400));
    let env = env_map(&[("IGNIS_SERVER_STREAM_TIMEOUT", "90")]);
    assert_eq!(expect_config(resolve(&[], env).expect("resolve")).stream_timeout_secs, 90);
    let env = env_map(&[("IGNIS_SERVER_STREAM_TIMEOUT", "90")]);
    let zero = expect_config(resolve(&args(&["--server-stream-timeout", "0", "--server-request-timeout", "0"]), env).expect("resolve"));
    assert_eq!((zero.stream_timeout_secs, zero.request_timeout_secs), (0, 0), "flag must win over env, and 0 is no deadline");
    let err = refused(&["--server-stream-timeout", "3601"]);
    assert!(err.contains("0..=3600") && err.contains("3601"), "{err}");
    for name in ["request_timeout", "stream_timeout"] {
        let field = schema::field("server", name).expect("declared");
        assert!(field.visible && field.patchable && !field.reload_required, "{field:?}");
    }
    let files = MemFiles::with(&[("c.yaml", "server:\n  stream_timeout: 45\n")]);
    assert_eq!(with_files(&["--config", "c.yaml"], no_env, &files).unwrap().stream_timeout_secs, 45, "the file key is server.stream_timeout");
    assert!(help().contains("--server-stream-timeout"));
    let live = crate::Live::of(&config(&["--server-request-timeout", "30", "--server-stream-timeout", "0"]));
    assert_eq!((live.request_timeout, live.stream_timeout), (std::time::Duration::from_secs(30), std::time::Duration::ZERO));
    let built = crate::Live::default();
    assert_eq!((built.request_timeout, built.stream_timeout), (std::time::Duration::from_secs(2400), std::time::Duration::from_secs(2400)));
}

/// `server.timeout_reply` says what a passed deadline answers: `partial`
/// (the default) or `error`, any case, refused otherwise naming both; live.
#[test]
fn the_timeout_reply_is_partial_by_default_or_error() {
    use crate::deadline::TimeoutReply;
    assert_eq!(config(&[]).timeout_reply, TimeoutReply::Partial);
    let env = env_map(&[("IGNIS_SERVER_TIMEOUT_REPLY", "Error")]);
    assert_eq!(expect_config(resolve(&[], env).expect("resolve")).timeout_reply, TimeoutReply::Error);
    let err = refused(&["--server-timeout-reply", "truncate"]);
    assert!(err.contains("partial") && err.contains("error") && err.contains("truncate"), "{err}");
    let field = schema::field("server", "timeout_reply").expect("declared");
    assert!(field.visible && field.patchable && !field.reload_required, "{field:?}");
    let files = MemFiles::with(&[("c.yaml", "server:\n  timeout_reply: error\n")]);
    assert_eq!(with_files(&["--config", "c.yaml"], no_env, &files).unwrap().timeout_reply, TimeoutReply::Error);
    assert_eq!(crate::Live::of(&config(&["--server-timeout-reply", "error"])).timeout_reply, TimeoutReply::Error);
    assert_eq!(crate::Live::default().timeout_reply, TimeoutReply::Partial);
}

// ── the model switch (spec model-switch/01) ──────────────────────────────

#[test]
fn the_switch_drain_timeout_defaults_to_thirty_seconds_takes_zero_and_has_a_ceiling() {
    assert_eq!(config(&[]).switch_drain_timeout_secs, 30);
    let env = env_map(&[("IGNIS_SWITCH_DRAIN_TIMEOUT", "90")]);
    assert_eq!(expect_config(resolve(&[], env).expect("resolve")).switch_drain_timeout_secs, 90);
    let env = env_map(&[("IGNIS_SWITCH_DRAIN_TIMEOUT", "90")]);
    let zero = expect_config(resolve(&args(&["--switch-drain-timeout", "0"]), env).expect("resolve"));
    assert_eq!(zero.switch_drain_timeout_secs, 0, "flag must win over env, and 0 cuts at once");
    let err = refused(&["--switch-drain-timeout", "3601"]);
    assert!(err.contains("3600") && err.contains("3601"), "{err}");
}

// ── idle unload (spec model-switch/02, GitHub #317) ──────────────────────

/// AC 1: `model.idle_unload` resolves from flag, env and file with the usual
/// precedence, defaults to five minutes, takes 0 (never), refuses a value
/// past its ceiling naming both, and is visible and patchable with no reload.
#[test]
fn the_idle_unload_defaults_to_five_minutes_takes_zero_and_has_a_ceiling() {
    assert_eq!(config(&[]).idle_unload_secs, 300);
    assert_eq!(DEFAULT_MODEL_IDLE_UNLOAD_SECS, 300);
    let env = env_map(&[("IGNIS_MODEL_IDLE_UNLOAD", "90")]);
    assert_eq!(expect_config(resolve(&[], env).expect("resolve")).idle_unload_secs, 90);
    let env = env_map(&[("IGNIS_MODEL_IDLE_UNLOAD", "90")]);
    let zero = expect_config(resolve(&args(&["--model-idle-unload", "0"]), env).expect("resolve"));
    assert_eq!(zero.idle_unload_secs, 0, "flag must win over env, and 0 never unloads");
    let ceiling = MAX_MODEL_IDLE_UNLOAD_SECS.to_string();
    let past = (MAX_MODEL_IDLE_UNLOAD_SECS + 1).to_string();
    let err = refused(&["--model-idle-unload", &past]);
    assert!(err.contains(&ceiling) && err.contains(&past), "{err}");
    let field = schema::field("model", "idle_unload").expect("declared");
    assert!(field.visible && field.patchable && !field.reload_required, "{field:?}");
    let files = MemFiles::with(&[("c.yaml", "model:\n  idle_unload: 45\n")]);
    assert_eq!(with_files(&["--config", "c.yaml"], no_env, &files).unwrap().idle_unload_secs, 45, "the file key is model.idle_unload");
}

// ── the Responses socket's idle close (GitHub #329) ──────────────────────

/// `server.socket_idle_close` resolves from flag, env and file with the
/// usual precedence, defaults to five minutes, takes 0 (never), refuses a
/// value past its ceiling naming both, and is visible and patchable with no
/// reload: the live knob a socket reads.
#[test]
fn the_socket_idle_close_defaults_to_five_minutes_takes_zero_and_has_a_ceiling() {
    assert_eq!(config(&[]).socket_idle_close_secs, 300);
    assert_eq!(DEFAULT_SOCKET_IDLE_CLOSE_SECS, 300);
    let env = env_map(&[("IGNIS_SERVER_SOCKET_IDLE_CLOSE", "90")]);
    assert_eq!(expect_config(resolve(&[], env).expect("resolve")).socket_idle_close_secs, 90);
    let env = env_map(&[("IGNIS_SERVER_SOCKET_IDLE_CLOSE", "90")]);
    let zero = expect_config(resolve(&args(&["--server-socket-idle-close", "0"]), env).expect("resolve"));
    assert_eq!(zero.socket_idle_close_secs, 0, "flag must win over env, and 0 never closes");
    let ceiling = MAX_SOCKET_IDLE_CLOSE_SECS.to_string();
    let past = (MAX_SOCKET_IDLE_CLOSE_SECS + 1).to_string();
    let err = refused(&["--server-socket-idle-close", &past]);
    assert!(err.contains(&ceiling) && err.contains(&past), "{err}");
    let field = schema::field("server", "socket_idle_close").expect("declared");
    assert!(field.visible && field.patchable && !field.reload_required, "{field:?}");
    let files = MemFiles::with(&[("c.yaml", "server:\n  socket_idle_close: 45\n")]);
    assert_eq!(
        with_files(&["--config", "c.yaml"], no_env, &files).unwrap().socket_idle_close_secs,
        45,
        "the file key is server.socket_idle_close"
    );
    let live = crate::Live::of(&config(&["--server-socket-idle-close", "30"]));
    assert_eq!(live.socket_idle_close, std::time::Duration::from_secs(30));
    assert_eq!(crate::Live::default().socket_idle_close, std::time::Duration::ZERO, "a server built without a config never closes");
}

// ── loop guard (ADR 0052) ────────────────────────────────────────────────

/// The loop guard scores normalized reasoning at 1,000 characters and 0.265
/// by default; `model.loop_guard_normalize` resolves from flag, env and
/// file, and the scoring knobs are visible and patchable with no reload.
#[test]
fn the_loop_guard_scores_normalized_by_default_and_off_scores_the_text_as_written() {
    let defaults = config(&[]).loop_guard;
    assert_eq!((defaults.normalize, defaults.window_chars, defaults.ratio_permille), (true, 1000, 265));
    assert_eq!(defaults, crate::loop_guard::LoopGuardParams::default());
    assert!(!config(&["--model-loop-guard-normalize", "off"]).loop_guard.normalize);
    let env = env_map(&[("IGNIS_MODEL_LOOP_GUARD_NORMALIZE", "false")]);
    assert!(!expect_config(resolve(&[], env).expect("resolve")).loop_guard.normalize);
    let env = env_map(&[("IGNIS_MODEL_LOOP_GUARD_NORMALIZE", "off")]);
    let flag = expect_config(resolve(&args(&["--model-loop-guard-normalize", "on"]), env).expect("resolve"));
    assert!(flag.loop_guard.normalize, "flag must win over env");
    assert!(refused(&["--model-loop-guard-normalize", "maybe"]).contains("maybe"));
    for name in ["loop_guard_normalize", "loop_guard_window", "loop_guard_ratio"] {
        let field = schema::field("model", name).expect("declared");
        assert!(field.visible && field.patchable && !field.reload_required, "{field:?}");
    }
    let files = MemFiles::with(&[("c.yaml", "model:\n  loop_guard_normalize: false\n  loop_guard_window: 3000\n  loop_guard_ratio: 120\n")]);
    let prototype = with_files(&["--config", "c.yaml"], no_env, &files).unwrap().loop_guard;
    assert_eq!((prototype.normalize, prototype.window_chars, prototype.ratio_permille), (false, 3000, 120), "the prototype's raw score");
}

/// The raw gate (0.30, 0 = none) and the clean window a rewind goes back to
/// (0.40, no margin) resolve from flag, env and file, visible and patchable
/// with no reload.
#[test]
fn the_loop_guards_raw_gate_and_clean_window_resolve_from_flag_env_and_file() {
    let defaults = config(&[]).loop_guard;
    assert_eq!((defaults.raw_gate_permille, defaults.clean_permille, defaults.margin_chars), (300, 400, 0));
    let flags = config(&["--model-loop-guard-raw-gate", "0", "--model-loop-guard-clean", "350"]).loop_guard;
    assert_eq!((flags.raw_gate_permille, flags.clean_permille), (0, 350), "0 turns the gate off");
    let env = env_map(&[("IGNIS_MODEL_LOOP_GUARD_RAW_GATE", "250"), ("IGNIS_MODEL_LOOP_GUARD_CLEAN", "450")]);
    let from_env = expect_config(resolve(&[], env).expect("resolve")).loop_guard;
    assert_eq!((from_env.raw_gate_permille, from_env.clean_permille), (250, 450));
    assert!(refused(&["--model-loop-guard-raw-gate", "1000"]).contains("--model-loop-guard-raw-gate"));
    assert!(refused(&["--model-loop-guard-clean", "0"]).contains("--model-loop-guard-clean"));
    for name in ["loop_guard_raw_gate", "loop_guard_clean", "loop_guard_margin"] {
        let field = schema::field("model", name).expect("declared");
        assert!(field.visible && field.patchable && !field.reload_required, "{field:?}");
    }
    let files = MemFiles::with(&[("c.yaml", "model:\n  loop_guard_raw_gate: 320\n  loop_guard_clean: 380\n  loop_guard_margin: 200\n")]);
    let file = with_files(&["--config", "c.yaml"], no_env, &files).unwrap().loop_guard;
    assert_eq!((file.raw_gate_permille, file.clean_permille, file.margin_chars), (320, 380, 200));
}

/// The guard rewinds by default (ADR 0052, the owner's 2026-10-14 call) and
/// `off` and `log` stay; a server built without a config runs with none, as
/// it runs with no idle unload.
#[test]
fn the_loop_guard_rewinds_by_default_and_off_and_log_stay() {
    use crate::loop_guard::LoopGuardMode;
    assert_eq!(config(&[]).loop_guard.mode, LoopGuardMode::Rewind);
    assert_eq!(crate::Live::of(&config(&[])).loop_guard.mode, LoopGuardMode::Rewind);
    assert_eq!(crate::Live::default().loop_guard.mode, LoopGuardMode::Off, "no in-process harness meets a rewind it did not ask for");
    assert_eq!(crate::Live::default().loop_guard, crate::loop_guard::LoopGuardParams { mode: LoopGuardMode::Off, ..config(&[]).loop_guard });
    for (raw, mode) in [("off", LoopGuardMode::Off), ("LOG", LoopGuardMode::Log), ("rewind", LoopGuardMode::Rewind)] {
        assert_eq!(config(&["--model-loop-guard", raw]).loop_guard.mode, mode, "{raw}");
    }
    let env = env_map(&[("IGNIS_MODEL_LOOP_GUARD", "off")]);
    assert_eq!(expect_config(resolve(&[], env).expect("resolve")).loop_guard.mode, LoopGuardMode::Off);
    assert!(refused(&["--model-loop-guard", "on"]).contains("on"));
}

/// The holdback: none by default, from flag, env and file, visible and
/// patchable with no reload, and bounded like the other character fields.
#[test]
fn the_loop_guards_holdback_is_off_by_default_and_resolves_from_flag_env_and_file() {
    assert_eq!(config(&[]).loop_guard.holdback_chars, 0);
    assert_eq!(config(&["--model-loop-guard-holdback", "3000"]).loop_guard.holdback_chars, 3000);
    let env = env_map(&[("IGNIS_MODEL_LOOP_GUARD_HOLDBACK", "6000")]);
    assert_eq!(expect_config(resolve(&[], env).expect("resolve")).loop_guard.holdback_chars, 6000);
    assert!(refused(&["--model-loop-guard-holdback", "100001"]).contains("--model-loop-guard-holdback"));
    let field = schema::field("model", "loop_guard_holdback").expect("declared");
    assert!(field.visible && field.patchable && !field.reload_required, "{field:?}");
    let files = MemFiles::with(&[("c.yaml", "model:\n  loop_guard_holdback: 2500\n")]);
    assert_eq!(with_files(&["--config", "c.yaml"], no_env, &files).unwrap().loop_guard.holdback_chars, 2500);
}

/// The guard's cadence (a check every 250 characters), the cut's snap to a
/// line break (300), the room a continuation needs (512 tokens) and how
/// long one the engine refuses as full waits (60 s) are config fields too,
/// from flag, env and file, visible and patchable with no reload.
#[test]
fn the_loop_guards_cadence_snap_room_and_submit_wait_resolve_from_flag_env_and_file() {
    let defaults = config(&[]).loop_guard;
    assert_eq!(
        (defaults.check_every_chars, defaults.snap_chars, defaults.min_continuation_tokens, defaults.submit_wait_secs),
        (250, 300, 512, 60)
    );
    let flags = config(&["--model-loop-guard-check-every", "500", "--model-loop-guard-snap", "0"]).loop_guard;
    assert_eq!((flags.check_every_chars, flags.snap_chars), (500, 0), "0 never snaps");
    let env = env_map(&[("IGNIS_MODEL_LOOP_GUARD_MIN_CONTINUATION", "1024"), ("IGNIS_MODEL_LOOP_GUARD_SUBMIT_WAIT", "0")]);
    let from_env = expect_config(resolve(&[], env).expect("resolve")).loop_guard;
    assert_eq!((from_env.min_continuation_tokens, from_env.submit_wait_secs), (1024, 0), "0 tries once");
    assert!(refused(&["--model-loop-guard-check-every", "0"]).contains("--model-loop-guard-check-every"));
    assert!(refused(&["--model-loop-guard-submit-wait", "3601"]).contains("--model-loop-guard-submit-wait"));
    for name in ["loop_guard_check_every", "loop_guard_snap", "loop_guard_min_continuation", "loop_guard_submit_wait"] {
        let field = schema::field("model", name).expect("declared");
        assert!(field.visible && field.patchable && !field.reload_required, "{field:?}");
    }
    let files = MemFiles::with(&[(
        "c.yaml",
        "model:\n  loop_guard_check_every: 125\n  loop_guard_snap: 600\n  loop_guard_min_continuation: 256\n  loop_guard_submit_wait: 30\n",
    )]);
    let file = with_files(&["--config", "c.yaml"], no_env, &files).unwrap().loop_guard;
    assert_eq!((file.check_every_chars, file.snap_chars, file.min_continuation_tokens, file.submit_wait_secs), (125, 600, 256, 30));
}

/// A loop must persist 4 checks before the guard acts, and a rewind must
/// keep a character of reasoning, by default; both resolve from flag, env
/// and file, visible and patchable with no reload, and are bounded.
#[test]
fn the_loop_guards_persistence_and_minimum_kept_resolve_from_flag_env_and_file() {
    let defaults = config(&[]).loop_guard;
    assert_eq!((defaults.persist_checks, defaults.min_kept_chars), (4, 1));
    let flags = config(&["--model-loop-guard-persist", "1", "--model-loop-guard-min-kept", "0"]).loop_guard;
    assert_eq!((flags.persist_checks, flags.min_kept_chars), (1, 0), "today's first check, and rewinds from nothing");
    let env = env_map(&[("IGNIS_MODEL_LOOP_GUARD_PERSIST", "6"), ("IGNIS_MODEL_LOOP_GUARD_MIN_KEPT", "500")]);
    let from_env = expect_config(resolve(&[], env).expect("resolve")).loop_guard;
    assert_eq!((from_env.persist_checks, from_env.min_kept_chars), (6, 500));
    assert!(refused(&["--model-loop-guard-persist", "0"]).contains("--model-loop-guard-persist"));
    assert!(refused(&["--model-loop-guard-min-kept", "100001"]).contains("--model-loop-guard-min-kept"));
    for name in ["loop_guard_persist", "loop_guard_min_kept"] {
        let field = schema::field("model", name).expect("declared");
        assert!(field.visible && field.patchable && !field.reload_required, "{field:?}");
    }
    let files = MemFiles::with(&[("c.yaml", "model:\n  loop_guard_persist: 3\n  loop_guard_min_kept: 200\n")]);
    let file = with_files(&["--config", "c.yaml"], no_env, &files).unwrap().loop_guard;
    assert_eq!((file.persist_checks, file.min_kept_chars), (3, 200));
}

/// The two nudges are the model's own words by default and config fields
/// like the rest, from flag, env and file, visible and patchable with no
/// reload.
#[test]
fn the_loop_guards_nudges_resolve_from_flag_env_and_file() {
    use crate::loop_guard::{DEFAULT_NUDGE, DEFAULT_NUDGE_AGAIN};
    let defaults = config(&[]).loop_guard;
    assert_eq!((defaults.nudge.as_str(), defaults.nudge_again.as_str()), (DEFAULT_NUDGE, DEFAULT_NUDGE_AGAIN));
    let flags = config(&["--model-loop-guard-nudge", "Wait, I'm repeating myself. Let me go on to the next step."]).loop_guard;
    assert_eq!(flags.nudge, "Wait, I'm repeating myself. Let me go on to the next step.");
    assert_eq!(flags.nudge_again, DEFAULT_NUDGE_AGAIN, "each has its own field");
    let env = env_map(&[("IGNIS_MODEL_LOOP_GUARD_NUDGE_AGAIN", "Enough. Wrap up.")]);
    assert_eq!(expect_config(resolve(&[], env).expect("resolve")).loop_guard.nudge_again, "Enough. Wrap up.");
    for name in ["loop_guard_nudge", "loop_guard_nudge_again"] {
        let field = schema::field("model", name).expect("declared");
        assert!(field.visible && field.patchable && !field.reload_required, "{field:?}");
    }
    let files = MemFiles::with(&[("c.yaml", "model:\n  loop_guard_nudge: \"Next step.\"\n  loop_guard_nudge_again: \"Wrap up.\"\n")]);
    let file = with_files(&["--config", "c.yaml"], no_env, &files).unwrap().loop_guard;
    assert_eq!((file.nudge.as_str(), file.nudge_again.as_str()), ("Next step.", "Wrap up."));
}

#[test]
fn a_model_field_may_switch_models_unless_the_operator_says_otherwise() {
    assert!(config(&[]).allow_model_switch);
    for (raw, on) in [("true", true), ("1", true), ("ON", true), ("false", false), ("0", false), ("off", false)] {
        assert_eq!(config(&["--switch-allow-implicit", raw]).allow_model_switch, on, "{raw}");
    }
    let env = env_map(&[("IGNIS_SWITCH_ALLOW_IMPLICIT", "off")]);
    assert!(!expect_config(resolve(&[], env).expect("resolve")).allow_model_switch);
    assert!(refused(&["--switch-allow-implicit", "maybe"]).contains("maybe"));
}

#[test]
fn known_models_come_from_repeated_flags_or_the_env_var_and_a_flag_replaces_the_env() {
    let a = args(&[
        "--switch-known-models",
        "qwen3.8-flash-next=F:/models/Qwen3.8-Flash-Next.ninfer",
        "--switch-known-models",
        " qwen3.8-27b = F:\\models\\Qwen3.8-27B.ninfer ",
    ]);
    let config = expect_config(resolve(&a, no_env).expect("resolve"));
    assert_eq!(
        config.known_models,
        BTreeMap::from([
            ("qwen3.8-flash-next".to_owned(), PathBuf::from("F:/models/Qwen3.8-Flash-Next.ninfer")),
            ("qwen3.8-27b".to_owned(), PathBuf::from("F:\\models\\Qwen3.8-27B.ninfer")),
        ])
    );
    let env = env_map(&[("IGNIS_SWITCH_KNOWN_MODELS", "qwen3.8-27b=F:/27b.ninfer;qwen3.8-flash-next=F:/fn.ninfer")]);
    assert_eq!(expect_config(resolve(&[], &env).expect("resolve")).known_models.len(), 2);
    let flag = expect_config(resolve(&args(&["--switch-known-models", "other=F:/other.ninfer"]), &env).expect("resolve"));
    assert_eq!(flag.known_models.keys().collect::<Vec<_>>(), ["other"], "a flag overrides its env var whole");
    let err = refused(&["--switch-known-models", "m=F:/a.ninfer", "--switch-known-models", "m=F:/b.ninfer"]);
    assert!(err.contains("`m`") && err.contains("twice"), "{err}");
}

// ── speculation as a load option (P5-02, GitHub #150) ────────────────────

#[test]
fn spec_dflash2_with_a_window_in_range_parses() {
    for n in 1..=7u32 {
        let config = config(&["--spec-backend", "dflash2", "--spec-draft-tokens", &n.to_string()]);
        assert_eq!(config.speculation, Some(Speculation::new(SpeculativeBackend::Dflash2, n).unwrap()));
    }
}

#[test]
fn a_draft_window_outside_1_to_7_is_refused_naming_the_flag_and_the_value() {
    for raw in ["0", "8", "15", "-1", "seven"] {
        let err = refused(&["--spec-backend", "dflash2", "--spec-draft-tokens", raw]);
        assert!(err.contains("--spec-draft-tokens") && err.contains(raw), "{err}");
    }
    assert!(refused(&["--spec-backend", "dflash2", "--spec-draft-tokens", "8"]).contains("1..=7"));
}

#[test]
fn an_unknown_speculative_backend_is_refused_naming_every_backend() {
    let err = refused(&["--spec-backend", "eagle", "--spec-draft-tokens", "3"]);
    assert!(err.contains("--spec-backend") && err.contains("eagle"), "{err}");
    assert!(err.contains("dflash2") && err.contains("off"), "{err}");
}

#[test]
fn spec_without_a_draft_window_is_refused() {
    let err = refused(&["--spec-backend", "dflash2"]);
    assert!(err.contains("--spec-draft-tokens") && err.contains("1..=7"), "{err}");
}

#[test]
fn a_draft_window_without_spec_is_refused_rather_than_ignored() {
    assert!(refused(&["--spec-draft-tokens", "7"]).contains("--spec-backend"));
}

#[test]
fn the_draft_head_defaults_to_full_and_takes_shortlist_by_flag_or_env() {
    let full = config(&["--spec-backend", "dflash2", "--spec-draft-tokens", "7"]);
    assert_eq!(full.speculation.map(|s| s.proposal_head()), Some(ProposalHead::Full));
    let short = config(&["--spec-backend", "dflash2", "--spec-draft-tokens", "7", "--spec-draft-head", "shortlist"]);
    assert_eq!(short.speculation.map(|s| s.proposal_head()), Some(ProposalHead::Shortlist));
    assert_eq!(short.speculation.map(|s| s.draft_tokens()), Some(7));
    let env = env_map(&[("IGNIS_SPEC_BACKEND", "dflash2"), ("IGNIS_SPEC_DRAFT_TOKENS", "7"), ("IGNIS_SPEC_DRAFT_HEAD", "shortlist")]);
    let from_env = expect_config(resolve(&[], env).expect("resolve"));
    assert_eq!(from_env.speculation.map(|s| s.proposal_head()), Some(ProposalHead::Shortlist));
}

/// ADR 0063: `auto` is a third spelling of the head, which the config keeps
/// as given and the load's speculation resolves to the full head -- with
/// DFlash2, and with MTP, whose only head it is.
#[test]
fn the_draft_head_auto_is_kept_as_configured_and_resolves_to_full() {
    let auto = config(&["--spec-backend", "dflash2", "--spec-draft-tokens", "7", "--spec-draft-head", "auto"]);
    assert_eq!(auto.settings().spec.draft_head, Some(ProposalHeadSetting::Auto));
    assert_eq!(auto.speculation.map(|s| s.proposal_head()), Some(ProposalHead::Full));
    // Without a backend there is still nothing to choose.
    let err = refused(&["--spec-draft-head", "auto"]);
    assert!(err.contains("--spec-draft-head auto") && err.contains("--spec-backend"), "{err}");
}

#[test]
fn a_draft_head_without_spec_or_with_an_unknown_name_is_refused() {
    let err = refused(&["--spec-draft-head", "shortlist"]);
    assert!(err.contains("--spec-draft-head") && err.contains("--spec-backend"), "{err}");
    let err = refused(&["--spec-backend", "dflash2", "--spec-draft-tokens", "7", "--spec-draft-head", "tiny"]);
    assert!(err.contains("--spec-draft-head") && err.contains("tiny"), "{err}");
}

#[test]
fn mtp_takes_no_proposal_head_and_spec_off_takes_no_draft_options() {
    let err = refused(&["--spec-backend", "mtp", "--spec-draft-head", "shortlist"]);
    assert!(err.contains("dflash2, off") && err.contains("mtp"), "{err}");

    let err = refused(&["--spec-backend", "off", "--spec-draft-tokens", "7"]);
    assert!(err.contains("--spec-draft-tokens") && err.contains("off"), "{err}");
    assert!(refused(&["--spec-backend", "off", "--spec-draft-head", "shortlist"]).contains("--spec-draft-head"));
    assert!(resolve(&[], env_map(&[("IGNIS_SPEC_BACKEND", "off"), ("IGNIS_SPEC_DRAFT_TOKENS", "3")])).is_err());
    assert!(config(&["--spec-backend", "off"]).speculation.is_none());
}

#[test]
fn the_speculation_env_vars_apply_and_the_flags_win_over_them() {
    let env = env_map(&[("IGNIS_SPEC_BACKEND", "dflash2"), ("IGNIS_SPEC_DRAFT_TOKENS", "3")]);
    assert_eq!(expect_config(resolve(&[], env).expect("resolve")).speculation.map(|s| s.draft_tokens()), Some(3));
    let env = env_map(&[("IGNIS_SPEC_BACKEND", "dflash2"), ("IGNIS_SPEC_DRAFT_TOKENS", "3")]);
    let flag = expect_config(resolve(&args(&["--spec-draft-tokens", "7"]), env).expect("resolve"));
    assert_eq!(flag.speculation.map(|s| s.draft_tokens()), Some(7), "flag must win over env");
}

#[test]
fn help_lists_the_speculation_flags() {
    let text = help();
    assert!(text.contains("--spec-backend") && text.contains("--spec-draft-tokens"), "{text}");
}

// ── rope scaling as a load option (GitHub #227) ──────────────────────────

#[test]
fn rope_scaling_is_auto_by_default_and_none_at_the_default_context() {
    let config = config(&["--profile", "none"]);
    assert!(config.settings().model.rope_scaling.is_auto());
    assert_eq!(config.rope_scaling, RopeScaling::NONE);
    assert!(!config.rope_scaling.is_yarn());
}

#[test]
fn auto_follows_the_context_and_an_explicit_value_overrides_it() {
    for (context, factor) in [("524288", 2.0), ("786432", 3.0), ("1048576", 4.0), ("589824", 2.25)] {
        let auto = config(&["--profile", "none", "--model-max-context", context]);
        assert!(auto.settings().model.rope_scaling.is_auto() && auto.rope_scaling.is_yarn(), "{context}");
        assert_eq!(auto.rope_scaling.factor(), factor, "{context}");
    }
    let at_trained = config(&["--profile", "none", "--model-max-context", "262144"]);
    assert_eq!(at_trained.rope_scaling, RopeScaling::NONE);
    let explicit = config(&["--profile", "none", "--model-max-context", "1048576", "--model-rope-scaling", "yarn:2"]);
    assert!(!explicit.settings().model.rope_scaling.is_auto());
    assert_eq!(explicit.rope_scaling.factor(), 2.0);
    let none = config(&["--profile", "none", "--model-max-context", "1048576", "--model-rope-scaling", "none"]);
    assert_eq!(none.rope_scaling, RopeScaling::NONE);
    let spelled = config(&["--profile", "none", "--model-max-context", "1048576", "--model-rope-scaling", "auto"]);
    assert_eq!(spelled.rope_scaling.factor(), 4.0);
}

#[test]
fn the_rope_scaling_flag_and_env_carry_the_reference_grammar() {
    let yarn = config(&["--model-rope-scaling", "yarn:4,t=0.25"]);
    assert!(yarn.rope_scaling.is_yarn());
    assert_eq!(yarn.rope_scaling.factor(), 4.0);
    assert_eq!(yarn.rope_scaling.temperature(), 0.25);
    let env = env_map(&[("IGNIS_MODEL_ROPE_SCALING", "yarn:2")]);
    assert_eq!(expect_config(resolve(&[], env).expect("resolve")).rope_scaling.factor(), 2.0);
    let env = env_map(&[("IGNIS_MODEL_ROPE_SCALING", "yarn:2")]);
    let none = expect_config(resolve(&args(&["--model-rope-scaling", "none"]), env).expect("resolve"));
    assert_eq!(none.rope_scaling, RopeScaling::NONE, "flag must win over env");
}

#[test]
fn rope_scaling_auto_is_settable_from_a_file_and_a_patch_moves_the_factor_with_the_context() {
    let files = MemFiles::with(&[("c.yaml", "model:
  rope_scaling: auto
  max_context: 786432
")]);
    let from_file = with_files(&["--profile", "none", "--config", "c.yaml"], no_env, &files).unwrap();
    assert!(from_file.settings().model.rope_scaling.is_auto());
    assert_eq!(from_file.rope_scaling.factor(), 3.0);

    // A context change under `auto` re-resolves; both fields are ReloadRequired.
    assert!(schema::field("model", "max_context").unwrap().reload_required);
    assert!(schema::field("model", "rope_scaling").unwrap().reload_required);
    let wider = from_file.with_patch(&patch_of(&[("model", "max_context", "1048576")])).unwrap();
    assert_eq!(wider.rope_scaling.factor(), 4.0);
    // An explicit factor stays put under a smaller context; `auto` takes over again.
    let pinned = from_file.with_patch(&patch_of(&[("model", "rope_scaling", "yarn:2"), ("model", "max_context", "65536")])).unwrap();
    assert!(!pinned.settings().model.rope_scaling.is_auto());
    assert_eq!(pinned.rope_scaling.factor(), 2.0);
    let back = pinned.with_patch(&patch_of(&[("model", "rope_scaling", "auto")])).unwrap();
    assert!(back.settings().model.rope_scaling.is_auto());
    assert_eq!(back.rope_scaling, RopeScaling::NONE, "65,536 is inside the trained position envelope");
}

#[test]
fn a_bad_rope_scaling_is_a_startup_error_naming_the_flag() {
    assert!(refused(&["--model-rope-scaling", "yarn:0.5"]).contains("--model-rope-scaling"));
    let shape = refused(&["--model-rope-scaling", "linear"]);
    assert!(shape.contains("--model-rope-scaling") && shape.contains("`auto`, `none` or `yarn:F"), "{shape}");
}

/// The image model (ADR 0050) decodes no tokens: any speculative backend is
/// refused at start, saying it has no drafter.
#[test]
fn the_image_model_refuses_speculation_saying_it_has_no_drafter() {
    let spec = config(&["--spec-backend", "dflash2", "--spec-draft-tokens", "7"]);
    let err = served_model_for(&spec, ModelFamily::QwenImage21).expect_err("no drafter").0;
    assert!(err.contains("--spec-backend dflash2") && err.contains("Qwen-Image-2.1 has no drafter"), "{err}");
}

#[test]
fn the_27b_takes_every_start_option_it_took_before() {
    let config = config(&["--spec-backend", "dflash2", "--spec-draft-tokens", "7", "--vision-enabled"]);
    assert_eq!(served_model_for(&config, ModelFamily::Qwen38_27b), Ok(DEFAULT_MODEL.to_owned()));
}

/// One patch entry, as `PATCH /v1/config` would give it.
fn patch_of(entries: &[(&str, &str, &str)]) -> source::Layer {
    let mut patch = source::Layer::default();
    for (group, name, raw) in entries {
        let meta = schema::field(group, name).unwrap();
        patch.set(meta, None, source::Candidate { raw: raw.to_string(), spelling: meta.file_key() }).unwrap();
    }
    patch
}

// ── vision as a load option (GitHub #177) ────────────────────────────────

#[test]
fn vision_is_off_by_default() {
    assert_eq!(config(&["--profile", "none"]).vision, None);
}

#[test]
fn the_vision_flag_loads_the_default_envelope() {
    let config = config(&["--profile", "none", "--vision-enabled"]);
    assert_eq!(config.vision, Some(Vision::default()));
    assert_eq!(config.vision.unwrap().max_tokens(), DEFAULT_VISION_MAX_TOKENS);
}

#[test]
fn the_vision_envelope_can_be_lowered() {
    assert_eq!(config(&["--vision-enabled", "--vision-max-tokens", "8192"]).vision.map(|v| v.max_tokens()), Some(8192));
}

#[test]
fn a_vision_envelope_without_vision_is_refused_rather_than_ignored() {
    assert!(refused(&["--profile", "none", "--vision-max-tokens", "8192"]).contains("--vision-enabled"));
    assert!(
        resolve(&args(&["--profile", "none"]), env_map(&[("IGNIS_VISION_MAX_TOKENS", "8192")])).is_err(),
        "the env form too"
    );
}

#[test]
fn the_embedding_pool_defaults_to_one_envelope_wide_item() {
    // GitHub #243 must not move the VRAM plan for an operator who does not
    // ask: the default pool is the reservation vision always took.
    let vision = config(&["--vision-enabled"]).vision.expect("vision on");
    assert_eq!(vision.pool_bytes(262_144), vision.output_transient_bytes(262_144));
}

#[test]
fn the_embedding_pool_can_be_widened() {
    let vision = config(&["--vision-enabled", "--vision-embedding-pool-mib", "1280"]).vision.expect("vision on");
    assert_eq!(vision.pool_bytes(262_144), 1280 * 1024 * 1024);
    let env = env_map(&[("IGNIS_VISION_ENABLED", "1"), ("IGNIS_VISION_EMBEDDING_POOL_MIB", "640")]);
    assert_eq!(expect_config(resolve(&[], env).expect("resolve")).vision.unwrap().pool_bytes(262_144), 640 * 1024 * 1024);
}

#[test]
fn an_embedding_pool_below_the_envelope_is_raised_not_refused() {
    // An operator lowering the pool should not have to recompute the
    // envelope's bytes to keep the load working: the floor is applied, and
    // the plan reports what was actually reserved.
    let vision = config(&["--vision-enabled", "--vision-embedding-pool-mib", "1"]).vision.expect("vision on");
    assert_eq!(vision.requested_pool_bytes(), 1024 * 1024);
    assert_eq!(vision.pool_bytes(262_144), vision.output_transient_bytes(262_144));
}

#[test]
fn an_embedding_pool_without_vision_is_refused_rather_than_ignored() {
    assert!(refused(&["--profile", "none", "--vision-embedding-pool-mib", "640"]).contains("--vision-enabled"));
    assert!(
        resolve(&args(&["--profile", "none"]), env_map(&[("IGNIS_VISION_EMBEDDING_POOL_MIB", "640")])).is_err(),
        "the env form too"
    );
}

#[test]
fn an_embedding_pool_outside_the_range_is_refused_naming_it() {
    for raw in ["0", "65537", "-1", "lots"] {
        let err = refused(&["--vision-enabled", "--vision-embedding-pool-mib", raw]);
        assert!(err.contains("--vision-embedding-pool-mib") && err.contains(raw), "{err}");
    }
}

#[test]
fn a_vision_envelope_outside_the_range_is_refused_naming_it() {
    for raw in ["0", "1048577", "-1", "lots"] {
        let err = refused(&["--vision-enabled", "--vision-max-tokens", raw]);
        assert!(err.contains("--vision-max-tokens") && err.contains(raw), "{err}");
    }
    assert!(refused(&["--vision-enabled", "--vision-max-tokens", "1048577"]).contains("1048576"));
}

#[test]
fn the_vision_env_vars_apply_and_the_flags_win_over_them() {
    let env = env_map(&[("IGNIS_VISION_ENABLED", "true"), ("IGNIS_VISION_MAX_TOKENS", "4096")]);
    assert_eq!(expect_config(resolve(&[], env).expect("resolve")).vision.map(|v| v.max_tokens()), Some(4096));
    let env = env_map(&[("IGNIS_VISION_ENABLED", "true"), ("IGNIS_VISION_MAX_TOKENS", "4096")]);
    let flag = expect_config(resolve(&args(&["--vision-max-tokens", "2048"]), env).expect("resolve"));
    assert_eq!(flag.vision.map(|v| v.max_tokens()), Some(2048), "flag must win over env");
    assert_eq!(expect_config(resolve(&[], env_map(&[("IGNIS_VISION_ENABLED", "false")])).expect("resolve")).vision, None);
    let err = resolve(&[], env_map(&[("IGNIS_VISION_ENABLED", "maybe")])).expect_err("bad bool").0;
    assert!(err.contains("IGNIS_VISION_ENABLED") && err.contains("maybe"), "{err}");
}

#[test]
fn help_lists_the_vision_flags() {
    let text = help();
    assert!(text.contains("--vision-enabled ") && text.contains("--vision-max-tokens"), "{text}");
    // GitHub #248: the flag shrinks an image, and says so.
    assert!(text.contains("is shrunk to fit"), "{text}");
}

// ── the vision residency (GitHub #414, ADR 0075) ─────────────────────────

/// A load that names no residency takes `host` (ADR 0075, GitHub #416); the
/// other values still win from any source.
#[test]
fn the_vision_residency_defaults_to_the_host() {
    let vision = config(&["--profile", "none", "--vision-enabled"]).vision.expect("vision on");
    assert_eq!(vision.residency(), VisionResidency::Host);
    let meta = schema::field("vision", "residency").expect("declared");
    assert_eq!((meta.default)(), serde_json::json!("host"));
    let device = config(&["--profile", "none", "--vision-enabled", "--vision-residency", "device"]);
    assert_eq!(device.vision.map(|v| v.residency()), Some(VisionResidency::Device));
}

/// ADR 0063 as amended by ADR 0075 (GitHub #416): the 5090 takes `host`; the
/// 96 GB card, where tower and pool are ~0.8% of the KV pool and the host may
/// be short of RAM, takes `device`.
#[test]
fn the_built_in_profiles_pick_the_vision_residency() {
    let at = |profile: &str| config(&["--profile", profile, "--vision-enabled"]).vision.map(|v| v.residency());
    assert_eq!(at("rtx50-16gb"), Some(VisionResidency::Host));

    assert_eq!(at("rtx50-16gb"), Some(VisionResidency::Host), "GitHub #418: written in the profile");
    let flag = config(&["--profile", "rtx50-16gb", "--vision-residency", "host"]);
    assert_eq!(flag.vision.map(|v| v.residency()), Some(VisionResidency::Host), "a flag wins over the profile");
}

#[test]
fn the_vision_residency_is_set_by_flag_env_and_file() {
    let flag = config(&["--profile", "none", "--vision-enabled", "--vision-residency", "host"]);
    assert_eq!(flag.vision.map(|v| v.residency()), Some(VisionResidency::Host));
    let env = env_map(&[("IGNIS_VISION_ENABLED", "true"), ("IGNIS_VISION_RESIDENCY", "host")]);
    let from_env = expect_config(resolve(&args(&["--profile", "none"]), env).expect("resolve"));
    assert_eq!(from_env.vision.map(|v| v.residency()), Some(VisionResidency::Host));
    let env = env_map(&[("IGNIS_VISION_ENABLED", "true"), ("IGNIS_VISION_RESIDENCY", "host")]);
    let flag_wins = expect_config(resolve(&args(&["--profile", "none", "--vision-residency", "device"]), env).expect("resolve"));
    assert_eq!(flag_wins.vision.map(|v| v.residency()), Some(VisionResidency::Device), "flag must win over env");
    let files = MemFiles::with(&[("c.yaml", "vision:\n  enabled: true\n  residency: host\n")]);
    let from_file = with_files(&["--profile", "none", "--config", "c.yaml"], no_env, &files).unwrap();
    assert_eq!(from_file.vision.map(|v| v.residency()), Some(VisionResidency::Host));
    // The envelope is the residency's neighbour, set beside it.
    let both = config(&["--profile", "none", "--vision-enabled", "--vision-residency", "host", "--vision-max-tokens", "4096"]);
    assert_eq!(both.vision.map(|v| (v.residency(), v.max_tokens())), Some((VisionResidency::Host, 4096)));
}

#[test]
fn the_vision_residency_is_patched_and_takes_effect_at_the_next_load() {
    let meta = schema::field("vision", "residency").expect("declared");
    assert!(meta.visible && meta.patchable && meta.reload_required && !meta.scoped, "{meta:?}");
    // A patch reaches the running 27B, whose next load it is.
    let started = config(&["--profile", "none", "--vision-enabled"]).for_family(ModelFamily::Qwen38_27b).unwrap();
    let patched = started.with_patch(&patch_of(&[("vision", "residency", "device")])).unwrap();
    assert_eq!(patched.vision.map(|v| v.residency()), Some(VisionResidency::Device));
}

#[test]
fn a_vision_residency_without_vision_is_refused_rather_than_ignored() {
    let err = refused(&["--profile", "none", "--vision-residency", "device"]);
    assert!(err.contains("--vision-residency device") && err.contains("--vision-enabled"), "{err}");
    assert!(
        resolve(&args(&["--profile", "none"]), env_map(&[("IGNIS_VISION_RESIDENCY", "device")])).is_err(),
        "the env form too"
    );
    // Naming the default is harmless, as for the envelope.
    assert!(config(&["--profile", "none", "--vision-residency", "host"]).vision.is_none());
}

#[test]
fn a_vision_residency_outside_its_values_is_refused_naming_them() {
    for raw in ["cpu", "on-demand", "vram"] {
        let err = refused(&["--profile", "none", "--vision-enabled", "--vision-residency", raw]);
        assert!(err.contains("--vision-residency") && err.contains("host") && err.contains("device"), "{raw}: {err}");
    }
    let shouted = config(&["--profile", "none", "--vision-enabled", "--vision-residency", "HOST"]);
    assert_eq!(shouted.vision.map(|v| v.residency()), Some(VisionResidency::Host), "case-insensitive");
}

#[test]
fn help_lists_the_vision_residency_and_both_values() {
    let block = help_block("--vision-residency");
    assert!(block.contains("host") && block.contains("device"), "{block}");
}

/// GitHub #195: the two are independent load options again.
#[test]
fn vision_and_dflash2_resolve_together_as_two_independent_load_options() {
    let both = config(&["--profile", "none", "--vision-enabled", "--spec-backend", "dflash2", "--spec-draft-tokens", "4"]);
    assert_eq!(both.vision, Some(Vision::default()));
    assert_eq!(both.speculation, Speculation::new(SpeculativeBackend::Dflash2, 4).ok());
    let env = env_map(&[("IGNIS_VISION_ENABLED", "true"), ("IGNIS_SPEC_BACKEND", "dflash2"), ("IGNIS_SPEC_DRAFT_TOKENS", "4")]);
    let from_env = expect_config(resolve(&args(&["--profile", "none"]), env).expect("the env form too"));
    assert_eq!((from_env.vision, from_env.speculation), (both.vision, both.speculation));
    let vision_only = config(&["--profile", "none", "--vision-enabled"]);
    assert_eq!((vision_only.vision, vision_only.speculation), (both.vision, None));
    let spec_only = config(&["--profile", "none", "--spec-backend", "dflash2", "--spec-draft-tokens", "4"]);
    assert_eq!((spec_only.vision, spec_only.speculation), (None, both.speculation));
}

// ── media acquisition (GitHub #179) ──────────────────────────────────────

#[test]
fn media_defaults_to_no_private_network_and_a_one_gib_cache() {
    assert_eq!(config(&["--vision-enabled"]).media, MediaOptions { allow_private_network: false, cache_bytes: 1024 << 20 });
    assert_eq!(config(&[]).media, MediaOptions::default());
}

#[test]
fn media_flags_set_the_private_network_opt_in_and_the_cache() {
    let flags = config(&["--vision-enabled", "--media-allow-private-network", "--media-cache-mib", "0"]);
    assert_eq!(flags.media, MediaOptions { allow_private_network: true, cache_bytes: 0 });
    let env = env_map(&[("IGNIS_VISION_ENABLED", "true"), ("IGNIS_MEDIA_ALLOW_PRIVATE_NETWORK", "true"), ("IGNIS_MEDIA_CACHE_MIB", "64")]);
    let mixed = expect_config(resolve(&args(&["--media-cache-mib", "128"]), env).expect("resolve"));
    assert_eq!(mixed.media, MediaOptions { allow_private_network: true, cache_bytes: 128 << 20 });
}

#[test]
fn media_flags_without_vision_are_refused_rather_than_ignored() {
    for a in [&["--media-allow-private-network"][..], &["--media-cache-mib", "10"]] {
        let mut flags = vec!["--profile", "none"];
        flags.extend_from_slice(a);
        let err = refused(&flags);
        assert!(err.contains(a[0]) && err.contains("--vision-enabled"), "{err}");
    }
}

#[test]
fn a_media_cache_outside_the_range_is_refused_naming_it() {
    for raw in ["65537", "-1", "lots"] {
        let err = refused(&["--vision-enabled", "--media-cache-mib", raw]);
        assert!(err.contains("--media-cache-mib") && err.contains(raw), "{err}");
    }
    let env = env_map(&[("IGNIS_VISION_ENABLED", "true"), ("IGNIS_MEDIA_ALLOW_PRIVATE_NETWORK", "maybe")]);
    assert!(resolve(&[], env).expect_err("bad bool").0.contains("maybe"));
}

#[test]
fn help_lists_the_media_flags() {
    let text = help();
    assert!(text.contains("--media-allow-private-network") && text.contains("--media-cache-mib"), "{text}");
}

// ── the Playground (GitHub #163, ADR 0026) ───────────────────────────────

#[test]
fn the_playground_is_on_by_default_and_off_when_asked() {
    assert!(config(&[]).ui);
    assert!(config(&["--server-ui"]).ui);
    assert!(!config(&["--server-ui", "false"]).ui);
}

#[test]
fn a_bare_bool_flag_leaves_the_next_flag_alone() {
    // The next argument is parsed as a flag of its own, not as a value.
    for (argv, want) in [(&["--server-ui", "--server-bind", "b"][..], true), (&["--server-ui", "off", "--server-bind", "b"], false)] {
        let config = config(argv);
        assert_eq!(config.ui, want, "{argv:?}");
        assert_eq!(config.bind, "b", "{argv:?}");
    }
}

#[test]
fn the_ui_env_var_applies_and_the_flags_win_over_it() {
    assert!(!expect_config(resolve(&[], env_map(&[("IGNIS_SERVER_UI", "false")])).expect("resolve")).ui);
    assert!(expect_config(resolve(&args(&["--server-ui"]), env_map(&[("IGNIS_SERVER_UI", "off")])).expect("resolve")).ui);
    assert!(!expect_config(resolve(&args(&["--server-ui", "false"]), env_map(&[("IGNIS_SERVER_UI", "true")])).expect("resolve")).ui);
    let err = resolve(&[], env_map(&[("IGNIS_SERVER_UI", "maybe")])).expect_err("not a boolean").0;
    assert!(err.contains("IGNIS_SERVER_UI"), "{err}");
}

#[test]
fn help_lists_the_ui_flag() {
    assert!(help().contains("--server-ui"));
}

// ── fetching a missing model (GitHub #234) ───────────────────────────────

#[test]
fn model_downloads_are_on_by_default_and_land_in_models() {
    let config = config(&[]);
    assert!(config.model_download);
    assert_eq!(config.model_download_path, PathBuf::from(DEFAULT_MODEL_DOWNLOAD_PATH));
}

#[test]
fn the_model_download_flag_stands_bare_or_takes_a_value() {
    for (argv, want) in [(&["--download-enabled", "--server-bind", "b"][..], true), (&["--download-enabled", "false", "--server-bind", "b"], false)] {
        let config = config(argv);
        assert_eq!(config.model_download, want, "{argv:?}");
        assert_eq!(config.bind, "b", "{argv:?}");
    }
}

#[test]
fn the_model_download_env_vars_apply_and_the_flags_win_over_them() {
    let env = env_map(&[("IGNIS_DOWNLOAD_ENABLED", "false"), ("IGNIS_DOWNLOAD_PATH", "/srv/models")]);
    let from_env = expect_config(resolve(&[], env).expect("resolve"));
    assert!(!from_env.model_download);
    assert_eq!(from_env.model_download_path, PathBuf::from("/srv/models"));
    assert!(expect_config(resolve(&args(&["--download-enabled"]), env_map(&[("IGNIS_DOWNLOAD_ENABLED", "off")])).expect("resolve")).model_download);
    let env = env_map(&[("IGNIS_DOWNLOAD_PATH", "/srv/models")]);
    let flag = expect_config(resolve(&args(&["--download-path", "D:/weights"]), env).expect("resolve"));
    assert_eq!(flag.model_download_path, PathBuf::from("D:/weights"));
    let err = resolve(&[], env_map(&[("IGNIS_DOWNLOAD_ENABLED", "maybe")])).expect_err("not a boolean").0;
    assert!(err.contains("IGNIS_DOWNLOAD_ENABLED"), "{err}");
}

#[test]
fn an_empty_model_download_path_is_the_default_not_the_working_directory() {
    // An unset-looking env var (`IGNIS_DOWNLOAD_PATH=`) must not turn the
    // destination into `.`, where a 19 GB file would land wherever the
    // server happened to be started from.
    let config = expect_config(resolve(&[], env_map(&[("IGNIS_DOWNLOAD_PATH", "")])).expect("resolve"));
    assert_eq!(config.model_download_path, PathBuf::from(DEFAULT_MODEL_DOWNLOAD_PATH));
}

#[test]
fn help_lists_both_model_download_flags() {
    let text = help();
    assert!(text.contains("--download-enabled") && text.contains("--download-path"), "{text}");
}

/// Spec model-download/02 AC 4: the endpoint, the token and the catalog are
/// fields like any other — a flag, an env var and a file key each, the flag
/// winning — and the endpoint must be a URL the transfer can use.
#[test]
fn the_download_endpoint_token_and_catalog_are_declared_fields() {
    let defaults = config(&[]);
    assert_eq!(defaults.download_endpoint, DEFAULT_DOWNLOAD_ENDPOINT);
    assert_eq!((defaults.download_token, defaults.download_catalog), (None, None));
    let env = env_map(&[
        ("IGNIS_DOWNLOAD_ENDPOINT", "https://artifactory.example.com/api/huggingfaceml/hf"),
        ("IGNIS_DOWNLOAD_TOKEN", "tok-env"),
        ("IGNIS_DOWNLOAD_CATALOG", "/srv/acme.catalog.yaml"),
    ]);
    let from_env = expect_config(resolve(&[], &env).expect("resolve"));
    assert_eq!(from_env.download_endpoint, "https://artifactory.example.com/api/huggingfaceml/hf");
    assert_eq!(from_env.download_token.as_ref().map(ApiKey::as_str), Some("tok-env"));
    assert_eq!(from_env.download_catalog, Some(PathBuf::from("/srv/acme.catalog.yaml")));
    let flags = expect_config(resolve(&args(&["--download-endpoint", "http://10.0.0.5:8080", "--download-token", "tok-flag"]), &env).expect("resolve"));
    assert_eq!(flags.download_endpoint, "http://10.0.0.5:8080");
    assert_eq!(flags.download_token.as_ref().map(ApiKey::as_str), Some("tok-flag"));
    assert!(!format!("{flags:?}").contains("tok-flag"), "a config dumped into a log line never shows the token");
    let file = MemFiles::with(&[("c.yaml", "download:\n  endpoint: https://mirror.example.com\n  token: tok-file\n")]);
    let from_file = expect_config(resolve_with(&args(&["--config", "c.yaml"]), no_env, &file).expect("resolve"));
    assert_eq!(from_file.download_endpoint, "https://mirror.example.com");
    assert_eq!(from_file.download_token.as_ref().map(ApiKey::as_str), Some("tok-file"));
    for bad in ["ftp://mirror.example.com", "mirror.example.com", "https://", "ftp://user:hunter2@mirror.example.com"] {
        let err = refused(&["--download-endpoint", bad]);
        assert!(err.contains("--download-endpoint") && err.contains("http(s) URL"), "{bad}: {err}");
        assert!(!err.contains("hunter2"), "{bad}: the refusal echoes the URL: {err}");
    }
    // A credential, a query or a fragment would reach `GET /v1/config`, `config
    // print` and the logs: refused, and the refusal never repeats the secret.
    for bad in ["https://user:hunter2@mirror.example.com", "https://hunter2@mirror.example.com", "https://:hunter2@mirror.example.com", "https://mirror.example.com/hf?token=hunter2", "https://mirror.example.com/hf#hunter2"] {
        let err = refused(&["--download-endpoint", bad]);
        assert!(err.contains("--download-endpoint") && err.contains("username, password, query or fragment"), "{bad}: {err}");
        assert!(!err.contains("hunter2"), "{bad}: the refusal echoes the secret: {err}");
    }
}

/// Spec model-download/02 AC 3: a relative `download.catalog` in a config
/// file is read beside that file; from a flag or an env var, from the
/// working directory; an absolute one as it is. The setting keeps the path
/// as written.
#[test]
fn a_relative_catalog_path_resolves_against_the_config_file_that_set_it() {
    let files = MemFiles::with(&[("conf/ignis.config.yaml", "download:\n  catalog: acme.catalog.yaml\n")]);
    let from_file = expect_config(resolve_with(&args(&["--config", "conf/ignis.config.yaml"]), no_env, &files).expect("resolve"));
    assert_eq!(from_file.download_catalog, Some(PathBuf::from("conf").join("acme.catalog.yaml")));
    assert_eq!(from_file.settings().download.catalog, Some(PathBuf::from("acme.catalog.yaml")), "written as given");
    let flagged = expect_config(
        resolve_with(&args(&["--config", "conf/ignis.config.yaml", "--download-catalog", "acme.catalog.yaml"]), no_env, &files).expect("resolve"),
    );
    assert_eq!(flagged.download_catalog, Some(PathBuf::from("acme.catalog.yaml")));
    let from_env = expect_config(
        resolve_with(&args(&["--config", "conf/ignis.config.yaml"]), env_map(&[("IGNIS_DOWNLOAD_CATALOG", "acme.catalog.yaml")]), &files).expect("resolve"),
    );
    assert_eq!(from_env.download_catalog, Some(PathBuf::from("acme.catalog.yaml")));
    let absolute = std::env::temp_dir().join("acme.catalog.yaml");
    let text = format!("download:\n  catalog: '{}'\n", absolute.display());
    let files = MemFiles::with(&[("conf/ignis.config.yaml", &text)]);
    let config = expect_config(resolve_with(&args(&["--config", "conf/ignis.config.yaml"]), no_env, &files).expect("resolve"));
    assert_eq!(config.download_catalog, Some(absolute));
}

// ── Prometheus metrics (GitHub #89, ADR 0017) ────────────────────────────

/// On by default since 2026-10-15 (the owner's call, ADR 0017), on their
/// own loopback listener; each of the three spellings turns them off.
#[test]
fn metrics_are_on_by_default_on_their_own_listener_and_off_when_asked() {
    assert!(DEFAULT_METRICS);
    assert_eq!(config(&[]).metrics, Some(DEFAULT_METRICS_BIND.to_owned()));
    assert_eq!(config(&["--server-metrics"]).metrics, Some(DEFAULT_METRICS_BIND.to_owned()));
    assert_ne!(DEFAULT_METRICS_BIND, DEFAULT_BIND);
    assert!(DEFAULT_METRICS_BIND.starts_with("127.0.0.1:"), "loopback, never exposed");
    for off in ["false", "off", "0"] {
        assert_eq!(config(&["--server-metrics", off]).metrics, None, "{off}");
    }
    let env = env_map(&[("IGNIS_SERVER_METRICS", "false")]);
    assert_eq!(expect_config(resolve(&[], env).expect("resolve")).metrics, None);
    let files = MemFiles::with(&[("c.yaml", "server:\n  metrics: false\n")]);
    assert_eq!(with_files(&["--config", "c.yaml"], no_env, &files).unwrap().metrics, None);
}

#[test]
fn metrics_bind_moves_the_metrics_listener_in_either_order_or_alone() {
    for argv in [
        &["--server-metrics", "--server-metrics-bind", "127.0.0.1:9100"][..],
        &["--server-metrics-bind", "127.0.0.1:9100", "--server-metrics"],
        &["--server-metrics-bind", "127.0.0.1:9100"],
    ] {
        assert_eq!(config(argv).metrics.as_deref(), Some("127.0.0.1:9100"), "{argv:?}");
    }
}

/// A default address another process holds (a second server on the host)
/// costs a start its listener, never the start; one the operator named is
/// theirs, and refuses it (ADR 0017's 2026-10-15 amendment). Named means an
/// address other than the default, from any of the three spellings.
#[test]
fn only_a_metrics_address_other_than_the_default_counts_as_named() {
    assert!(!config(&[]).metrics_bind_named);
    assert!(!config(&["--server-metrics"]).metrics_bind_named);
    assert!(!config(&["--server-metrics-bind", DEFAULT_METRICS_BIND]).metrics_bind_named, "the default, named");
    assert!(config(&["--server-metrics-bind", "127.0.0.1:9100"]).metrics_bind_named);
    let env = env_map(&[("IGNIS_SERVER_METRICS_BIND", "127.0.0.1:9100")]);
    assert!(expect_config(resolve(&[], env).expect("resolve")).metrics_bind_named);
    let files = MemFiles::with(&[("c.yaml", "server:\n  metrics_bind: 127.0.0.1:9100\n")]);
    assert!(with_files(&["--config", "c.yaml"], no_env, &files).unwrap().metrics_bind_named);
}

#[test]
fn metrics_bind_with_metrics_off_or_without_value_or_on_the_api_address_is_refused() {
    let err = refused(&["--server-metrics", "false", "--server-metrics-bind", "127.0.0.1:9100"]);
    assert!(err.contains("`--server-metrics-bind`") && err.contains("(`--server-metrics` is false)"), "{err}");
    assert!(resolve(&args(&["--server-metrics", "--server-metrics-bind"]), no_env).is_err());
    let err = refused(&["--server-metrics", "--server-bind", "127.0.0.1:7000", "--server-metrics-bind", "127.0.0.1:7000"]);
    assert!(err.contains("--server-bind"), "{err}");
}

/// The default metrics address is never a reason not to start: an API bound
/// there leaves the metrics listener to the warn path, with no flag named.
#[test]
fn an_api_on_the_default_metrics_address_starts_without_naming_a_metrics_flag() {
    let resolved = config(&["--server-bind", DEFAULT_METRICS_BIND]);
    assert_eq!(resolved.metrics.as_deref(), Some(DEFAULT_METRICS_BIND));
    assert!(!resolved.metrics_bind_named);
}

/// ADR 0046: every field has its env var, metrics included — the drift
/// (`IGNIS_PROMETHEUS` with no flag, `--metrics` with no env var) the one
/// declaration per field exists to end. The old names stay without effect.
#[test]
fn metrics_has_its_env_var_now_and_no_alias() {
    let env = env_map(&[("IGNIS_SERVER_METRICS", "true"), ("IGNIS_SERVER_METRICS_BIND", "127.0.0.1:9100")]);
    assert_eq!(expect_config(resolve(&[], env).expect("resolve")).metrics.as_deref(), Some("127.0.0.1:9100"));
    for name in ["IGNIS_METRICS", "IGNIS_PROMETHEUS", "IGNIS_METRICS_BIND"] {
        let env = move |key: &str| (key == name).then(|| "127.0.0.1:9100".to_owned());
        assert_eq!(expect_config(resolve(&[], env).expect("resolve")).metrics.as_deref(), Some(DEFAULT_METRICS_BIND), "{name}");
    }
    for alias in ["-M", "--prometheus", "--server-metrics=true", "--metrics"] {
        assert!(resolve(&args(&[alias]), no_env).is_err(), "`{alias}` is not a metrics alias");
    }
}

#[test]
fn help_lists_the_metrics_flags() {
    let text = help();
    assert!(text.contains("--server-metrics ") && text.contains("--server-metrics-bind"), "{text}");
}

// ── the API key ──────────────────────────────────────────────────────────

#[test]
fn the_api_key_is_unset_by_default_and_an_empty_value_stays_unset() {
    assert_eq!(config(&[]).api_key, None);
    assert_eq!(expect_config(resolve(&[], env_map(&[("IGNIS_SERVER_API_KEY", "")])).expect("resolve")).api_key, None);
}

#[test]
fn the_api_key_resolves_flag_over_env() {
    let env = env_map(&[("IGNIS_SERVER_API_KEY", "from-env")]);
    assert_eq!(expect_config(resolve(&[], &env).expect("resolve")).api_key, Some(ApiKeySetting::Fixed(ApiKey::new("from-env"))));
    let flag = expect_config(resolve(&args(&["--server-api-key", "from-flag"]), &env).expect("resolve"));
    assert_eq!(flag.api_key, Some(ApiKeySetting::Fixed(ApiKey::new("from-flag"))), "flag must win over env");
}

#[test]
fn auto_asks_for_a_generated_key_from_the_flag_or_the_env() {
    assert_eq!(config(&["--server-api-key", "auto"]).api_key, Some(ApiKeySetting::Generate));
    let env = env_map(&[("IGNIS_SERVER_API_KEY", "auto")]);
    assert_eq!(expect_config(resolve(&[], env).expect("resolve")).api_key, Some(ApiKeySetting::Generate));
}

#[test]
fn a_generated_key_is_fresh_and_long() {
    let a = ApiKey::generate().expect("random source");
    let b = ApiKey::generate().expect("random source");
    assert_ne!(a, b);
    let hex = a.as_str().strip_prefix("sk-ignis-").expect("prefix");
    assert_eq!(hex.len(), 64);
    assert!(hex.chars().all(|c| c.is_ascii_hexdigit()), "{hex}");
}

#[test]
fn an_api_key_matches_only_itself_and_never_prints() {
    let key = ApiKey::new("sk-secret");
    assert!(key.matches("sk-secret"));
    for other in ["", "sk-secre", "sk-secret!", "sk-Secret"] {
        assert!(!key.matches(other), "{other:?}");
    }
    let config = config(&["--server-api-key", "sk-secret"]);
    assert!(!format!("{config:?}").contains("sk-secret"));
    assert!(!format!("{:?}", config.for_family(ModelFamily::Qwen38_27b).unwrap()).contains("sk-secret"));
}

#[test]
fn help_lists_the_api_key_flag() {
    let text = help();
    assert!(text.contains("--server-api-key") && text.contains("IGNIS_SERVER_API_KEY"), "{text}");
}

// ── exposure (ADR 0028) ──────────────────────────────────────────────────

#[test]
fn nothing_is_exposed_by_default() {
    let config = config(&[]);
    assert_eq!(config.api_key, None, "no exposure, no forced key");
}

// ── `--profile auto` (ADR 0063, spec 27b-16gb/04) ────────────────────────

const GIB_F: f64 = (1u64 << 30) as f64;

fn on_card(flags: &[&str], card: profile::Card) -> Result<Config, ConfigError> {
    resolve_on(&args(flags), no_env, &file::NoFiles, &|| card.clone()).map(expect_config)
}

fn on_total(flags: &[&str], gib: f64) -> Result<Config, ConfigError> {
    on_card(flags, profile::Card::Total((gib * GIB_F) as u64))
}

/// Each card class gets its class's built-in profile, named by what it
/// resolved to, never by `auto`.
#[test]
fn auto_picks_the_profile_of_the_cards_class() {
    for (gib, name, class) in [
        (15.9, "rtx50-16gb", profile::CardClass::Gb16),


    ] {
        let config = on_total(&[], gib).unwrap();
        assert_eq!(config.basis.sources().profile_name, name, "{gib} GiB");
        assert!(config.basis.sources().profile_auto);
        assert_eq!(config.card_class(), Some(class));
        assert_eq!(config.card_total_bytes(), Some((gib * GIB_F) as u64));
        // `auto` spelled out is the default.
        assert_eq!(on_total(&["--profile", "auto"], gib).unwrap().basis.sources().profile_name, name);
    }
}

/// A total of no class refuses the start, naming the total and `--profile`
/// and the profiles there are.
#[test]
fn auto_refuses_a_card_of_no_class_naming_profile() {
    for gib in [24.0, 48.0] {
        let err = on_total(&[], gib).unwrap_err().0;
        assert!(err.contains(&format!("{gib:.1} GiB")), "{err}");
        assert!(err.contains("--profile") && err.contains("rtx50-16gb") && err.contains("rtx50-16gb"), "{err}");
    }
    let unreadable = on_card(&[], profile::Card::Unreadable("NVML is not installed".to_owned())).unwrap_err().0;
    assert!(unreadable.contains("--profile") && unreadable.contains("total"), "{unreadable}");
    assert!(unreadable.contains("NVML is not installed"), "the reason NVML failed is in the refusal: {unreadable}");
}

/// A build with no card (the mock engine, CI on Linux) resolves as `none`
/// would: the hardcoded defaults, said so by the profile's name.
#[test]
fn auto_with_no_card_resolves_as_none() {
    let absent = on_card(&[], profile::Card::Absent).unwrap();
    let none = on_card(&["--profile", "none"], profile::Card::Absent).unwrap();
    assert_eq!(absent.basis.sources().profile_name, "none");
    assert_eq!(absent.settings(), none.settings());
    assert_eq!(absent.card_class(), None);
}

/// A named profile always wins, `none` included, and no refusal of an odd
/// card reaches it; the total is read for the catalog's sake at a start.
#[test]
fn a_named_profile_beats_auto_whatever_the_card() {
    let flag = on_total(&["--profile", "rtx50-16gb"], 24.0).unwrap();
    assert_eq!(flag.basis.sources().profile_name, "rtx50-16gb");
    assert!(!flag.basis.sources().profile_auto);
    assert_eq!(flag.card_class(), None, "24 GiB is no class; the named profile still starts");
    assert_eq!(on_total(&["--profile", "none"], 31.8).unwrap().basis.sources().profile_name, "none");
    let env = |key: &str| (key == "IGNIS_PROFILE").then(|| "rtx50-16gb".to_owned());
    let from_env = resolve_on(&[], env, &file::NoFiles, &|| profile::Card::Total(profile::RTX_5080_TOTAL_BYTES)).map(expect_config).unwrap();
    assert_eq!(from_env.basis.sources().profile_name, "rtx50-16gb");
    assert_eq!(from_env.card_class(), Some(profile::CardClass::Gb16), "the total read is the card's");
    let files = MemFiles::with(&[("c.yaml", "profile: none\n")]);
    let from_file = resolve_on(&args(&["--config", "c.yaml"]), no_env, &files, &|| profile::Card::Total(24 << 30)).map(expect_config).unwrap();
    assert_eq!(from_file.basis.sources().profile_name, "none");
    // A named profile reads no total for the choice: a probe that panics
    // is never called by `gather`.
    let parsed = source::parse_args(&args(&["--profile", "rtx50-16gb"])).unwrap();
    gather(parsed, &no_env, &file::NoFiles, FileChoice::None, &|| panic!("a named profile reads no total")).unwrap();
}

/// AC 2's regression: on a 5090 `auto` resolves every field, for each
/// family, exactly as `--profile rtx50-16gb` does.
#[test]
fn auto_on_a_5080_resolves_exactly_as_the_named_profile_does() {
    let auto = on_card(&[], profile::Card::Total(profile::RTX_5080_TOTAL_BYTES)).unwrap();
    let named = config(&["--profile", "rtx50-16gb"]);
    assert_eq!(auto.settings(), named.settings());
    for family in field::FAMILIES {
        let (a, n) = (auto.for_family(family).unwrap(), named.for_family(family).unwrap());
        assert_eq!(a.settings(), n.settings(), "{family:?}");
        assert_eq!(a.max_context, n.max_context, "{family:?}");
    }
    assert_eq!(auto.for_family(ModelFamily::Qwen38_27b).unwrap().max_context, 262_144);
}

/// GitHub #418 (ADR 0063 as amended by ADR 0075): `rtx50-16gb` turns vision
/// on at a 4,096-token envelope, residency `host`; an operator's own
/// `--vision-enabled false` or `--vision-max-tokens` wins, and the 5090's
/// profile keeps the 32,768 default.
#[test]
fn rtx50_16gb_turns_vision_on_at_a_4096_token_envelope_and_the_operator_wins() {
    let envelope = |flags: &[&str], gib: f64| {
        let config = on_total(flags, gib).unwrap();
        config.for_family(ModelFamily::Qwen38_27b).unwrap().vision.map(|v| v.max_tokens())
    };
    assert_eq!(envelope(&[], 15.9), Some(4096), "on, at the profile's envelope");
    assert_eq!(envelope(&["--vision-enabled"], 15.9), Some(4096));
    assert_eq!(envelope(&["--vision-enabled", "false"], 15.9), None, "the operator's off wins");
    assert_eq!(envelope(&["--vision-max-tokens", "1024"], 15.9), Some(1024), "the operator's envelope wins");
    assert_eq!(envelope(&[], 15.9), Some(4096), "rtx50-16gb's vision is on at the default");
}

/// ADR 0063 §3: a value the profile or the hardcoded default gave adapts at
/// load; one the operator wrote -- by flag, env or file, even a file's value
/// at the hardcoded default -- never does, each of the three alone.
#[test]
fn only_the_profiles_and_the_defaults_values_adapt() {
    use crate::adaptation::Adaptable;
    let qwen = |config: Config| config.for_family(ModelFamily::Qwen38_27b).unwrap().adaptable();
    assert_eq!(qwen(config(&[])), Adaptable::ALL, "rtx50-16gb's context, the default lanes and head");
    assert_eq!(qwen(config(&["--model-max-context", "131072"])), Adaptable { max_context: false, ..Adaptable::ALL });
    let env = env_map(&[("IGNIS_QWEN38_SPEC_DECODE_LANES", "2")]);
    assert_eq!(qwen(expect_config(resolve(&[], env).unwrap())), Adaptable { decode_lanes: false, ..Adaptable::ALL });
    let files = MemFiles::with(&[
        ("head.yaml", "spec:\n  backend: dflash2\n  draft_tokens: 7\n  draft_head: auto\n"),
        ("default.yaml", "model:\n  max_context: 40960\n"),
    ]);
    let head = with_files(&["--config", "head.yaml"], no_env, &files).unwrap();
    assert_eq!(qwen(head), Adaptable { draft_head: false, ..Adaptable::ALL }, "an explicit `auto` too");
    let default = with_files(&["--config", "default.yaml"], no_env, &files).unwrap();
    assert_eq!(default.max_context, DEFAULT_MAX_CONTEXT);
    assert!(!qwen(default).max_context, "a file's value is explicit, at the default or not");
    // `--profile none`: the hardcoded defaults adapt as a profile's would.
    assert_eq!(qwen(config(&["--profile", "none"])), Adaptable::ALL);
}

/// ADR 0063 §3, review of #336 part D: the adaptation ladder's step and
/// spare reservation are fields, 65,536 tokens each unless named, by flag,
/// env or file; they reach the 27B's ladder.
#[test]
fn the_adaptation_ladder_is_two_fields_at_65536_tokens() {
    use crate::adaptation::Ladder;
    let defaults = config(&[]);
    assert_eq!((defaults.settings().model.adapt_context_step, defaults.settings().vram.adapt_spare_tokens), (65_536, 65_536));
    assert_eq!((DEFAULT_ADAPT_CONTEXT_STEP, DEFAULT_ADAPT_SPARE_TOKENS), (65_536, 65_536));
    assert_eq!(defaults.adapt_ladder(), Ladder::default());
    let named = config(&["--model-adapt-context-step", "131072", "--vram-adapt-spare-tokens", "0"]);
    assert_eq!(named.adapt_ladder(), Ladder { context_step: 131_072, spare_tokens: 0 }, "0: one sequence is enough");
    let env = env_map(&[("IGNIS_MODEL_ADAPT_CONTEXT_STEP", "4096"), ("IGNIS_VRAM_ADAPT_SPARE_TOKENS", "32768")]);
    let from_env = expect_config(resolve(&[], env).expect("resolve"));
    assert_eq!(from_env.adapt_ladder(), Ladder { context_step: 4_096, spare_tokens: 32_768 });
    let files = MemFiles::with(&[("c.yaml", "model:\n  adapt_context_step: 64\nvram:\n  adapt_spare_tokens: 1000\n")]);
    let from_file = with_files(&["--config", "c.yaml"], no_env, &files).unwrap();
    assert_eq!(from_file.adapt_ladder(), Ladder { context_step: 64, spare_tokens: 1_000 }, "one KV page is the smallest step");
}

/// The context step is a whole number of KV pages, at least one: anything
/// else is refused by name.
#[test]
fn an_adaptation_step_that_is_not_whole_kv_pages_is_refused() {
    for raw in ["0", "100", "65537", "-64", "wide"] {
        let err = refused(&["--model-adapt-context-step", raw]);
        assert!(err.contains("--model-adapt-context-step"), "{raw}: {err}");
    }
    assert!(refused(&["--model-adapt-context-step", "100"]).contains("nonzero multiple of 64"));
    assert!(refused(&["--vram-adapt-spare-tokens", "-1"]).contains("--vram-adapt-spare-tokens"));
}

/// The startup line says whether the profile was chosen by `auto` or named.
#[test]
fn the_startup_log_says_a_named_profile_is_named() {
    use tracing_subscriber::layer::SubscriberExt;
    let config = config(&["--profile", "rtx50-16gb"]);
    let sink = std::sync::Arc::new(ignis_logging::MemorySink::new());
    let subscriber = tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone()));
    tracing::subscriber::with_default(subscriber, || log_source(&config));
    let event: serde_json::Value = serde_json::from_str(&sink.lines()[0]).unwrap();
    let field = |name: &str| event.get(name).or_else(|| event["attributes"].get(name)).cloned();
    assert_eq!(field("profile"), Some("rtx50-16gb".into()), "{event}");
    assert_eq!(field("profile_by"), Some("named".into()), "{event}");
}

/// Help lists profiles available in this fork.
#[test]
fn help_lists_the_5080_profile_without_retired_options() {
    let help = help_text();
    assert!(help.contains("16 GB: rtx50-16gb"), "{help}");
    for retired in ["rtx5090", "rtxpro6000", "qwen38flashnext", "Flash-Next", "image job"] {
        assert!(!help.contains(retired), "{retired}: {help}");
    }
}

/// The refusal of a card of no class spells the boundaries the constants hold.
#[test]
fn the_refusal_of_a_card_of_no_class_spells_every_boundary() {
    let err = on_total(&[], 24.0).unwrap_err().0;
    for bounds in ["16 GB below 20 GiB", "32 GB from 28 to 40 GiB", "96 GB from 88 to 104 GiB"] {
        assert!(err.contains(bounds), "{bounds}: {err}");
    }
}

/// A `profiles:` entry called `none` or `auto` could never be reached
/// (`gather` resolves both words first): the file is refused when read.
#[test]
fn a_file_cannot_define_a_profile_called_none_or_auto() {
    for name in ["none", "auto"] {
        let contents = format!("profiles:\n  {name}:\n    vram:\n      headroom_bytes: 2G\n");
        let files = MemFiles::with(&[("c.yaml", contents.as_str())]);
        let err = resolve_with(&args(&["--config", "c.yaml"]), no_env, &files).map(expect_config).unwrap_err().0;
        assert!(err.contains(&format!("`{name}`")) && err.contains("c.yaml"), "{name}: {err}");
    }
}

/// `auto` picks by NAME through the lookup a named profile uses, so a file's
/// `profiles:` redefining `rtx50-16gb` replaces what `auto` picks for a 5090.
#[test]
fn a_file_redefining_a_built_in_profile_replaces_what_auto_picks() {
    let files = MemFiles::with(&[("c.yaml", "profiles:\n  rtx50-16gb:\n    model:\n      qwen38:\n        max_context: 131072\n")]);
    let config = resolve_on(&args(&["--config", "c.yaml"]), no_env, &files, &|| profile::Card::Total(profile::RTX_5080_TOTAL_BYTES))
        .map(expect_config)
        .unwrap();
    assert_eq!(config.basis.sources().profile_name, "rtx50-16gb");
    assert_eq!(config.for_family(ModelFamily::Qwen38_27b).unwrap().max_context, 131_072);
}

/// The `config` verbs resolve as a start would: on a card of no class,
/// `config print` under `auto` refuses naming `--profile`.
#[test]
fn the_config_verbs_refuse_a_card_of_no_class_under_auto() {
    let refused = resolve_on(&args(&["config", "print"]), no_env, &file::NoFiles, &|| profile::Card::Total(24 << 30)).unwrap_err().0;
    assert!(refused.contains("--profile"), "{refused}");
    resolve_on(&args(&["config", "print", "--profile", "none"]), no_env, &file::NoFiles, &|| profile::Card::Total(24 << 30)).unwrap();
}
