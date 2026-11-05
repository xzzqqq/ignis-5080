//! Hardware profiles (spec config-v2/02 §`--profile`): a named bundle of
//! field defaults for one kind of card, slotted between the config file and
//! the hardcoded defaults — a smarter default, never an override.
//!
//! The built-in profiles are written below in the config file's own
//! `profiles:` shape and read by the same [`super::file::read_document`] a
//! file's section is, so a profile an operator defines for a card the
//! maintainers do not own resolves exactly as a built-in one does: there is
//! one code path, and a built-in profile is only the default contents of it.

use std::collections::BTreeMap;

use super::file::read_document;
use super::source::Layer;
use super::ConfigError;

/// The profile used when nothing names one: [`AUTO_PROFILE`], which picks the
/// built-in profile by the card's total memory ([`CardClass::profile`]). An
/// unset `--profile` is this one, not "no profile" — for that, name
/// [`NONE_PROFILE`].
pub const DEFAULT_PROFILE: &str = AUTO_PROFILE;

/// `--profile auto` / `IGNIS_PROFILE=auto` (ADR 0063, spec 27b-16gb/04):
/// the built-in profile of the card's [`CardClass`], read from NVML's total
/// memory at the start. Not a name [`lookup`] resolves: `gather` chooses
/// the real profile first and names that one, so `GET /v1/config`, `make
/// config` and the UI show `rtx50-16gb`, never `auto`. It picks the
/// class's profile by NAME, through the same [`lookup`] a named profile goes
/// through: a config file's `profiles:` can redefine `rtx50-16gb`.
pub const AUTO_PROFILE: &str = "auto";

/// What the start could read of the card `--profile auto` chooses by. Injected
/// into config resolution, like the environment: resolution runs in contexts
/// with no card (CI on Linux without NVML, the CPU tests), so the module never
/// reads one itself. The `config` and `model` verbs read it exactly as a
/// start does (`config generate|print|patch`, `model list|download`), except
/// that the `model` verbs, which need no card, take [`Card::Unreadable`] for
/// [`Card::Absent`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Card {
    /// The card's total memory in bytes, as NVML reports it.
    Total(u64),
    /// A build with no CUDA (the mock engine, CI): there is no card to read,
    /// and `auto` resolves as `none` would — the hardcoded defaults.
    Absent,
    /// A CUDA build that could not read the total, and why (NVML's error):
    /// `auto` refuses, naming the reason and `--profile`.
    Unreadable(String),
}

impl Card {
    /// The card's total memory when it was read.
    pub fn total(&self) -> Option<u64> {
        match self {
            Self::Total(total) => Some(*total),
            Self::Absent | Self::Unreadable(_) => None,
        }
    }
}

/// The RTX 5080 total reported by NVML, injected by CPU config tests.
pub const RTX_5080_TOTAL_BYTES: u64 = 16_303 << 20;
#[cfg(test)]
pub const RTX_5090_TOTAL_BYTES: u64 = 32_607 << 20;

/// The profile `--profile auto` chooses for `card`: its class's built-in
/// profile (`Some(name)`), no profile at all on a build with no card
/// (`None`), or a refusal naming the total and `--profile`.
pub fn auto(card: Card) -> Result<Option<&'static str>, ConfigError> {
    match card {
        Card::Absent => Ok(None),
        Card::Unreadable(why) => Err(ConfigError(format!(
            "`--profile auto` could not read the card's total memory ({why}): name a profile with `--profile` or IGNIS_PROFILE ({}, or `none`)",
            built_in_names()
        ))),
        Card::Total(total) => CardClass::of_total(total).map(|class| Some(class.profile())).ok_or_else(|| {
            ConfigError(format!(
                "`--profile auto`: a card of {:.1} GiB is no card class ({}): name a profile with `--profile` or IGNIS_PROFILE ({}, or `none`)",
                total as f64 / (1u64 << 30) as f64,
                class_bounds(),
                built_in_names()
            ))
        }),
    }
}

/// The boundaries of every class, spelled from the `CARD_CLASS_*` constants:
/// "16 GB below 20 GiB, 32 GB from 28 to 40 GiB, 96 GB from 88 to 104 GiB".
fn class_bounds() -> String {
    let gib = |bytes: u64| bytes >> 30;
    CardClass::ALL
        .into_iter()
        .map(|class| match class {
            CardClass::Gb16 => format!("{class} below {} GiB", gib(CARD_CLASS_16GB_BELOW_BYTES)),
            CardClass::Gb32 => format!("{class} from {} to {} GiB", gib(*CARD_CLASS_32GB_BYTES.start()), gib(*CARD_CLASS_32GB_BYTES.end())),
            CardClass::Gb96 => format!("{class} from {} to {} GiB", gib(*CARD_CLASS_96GB_BYTES.start()), gib(*CARD_CLASS_96GB_BYTES.end())),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// The sentence `--help` gives `--profile auto`: each class's profile by its
/// own name, built from [`CardClass::ALL`] and [`CardClass::profile`].
pub fn classes_text() -> String {
    let profiles = built_in();
    CardClass::ALL.into_iter().filter(|class| profiles.contains_key(class.profile()))
        .map(|class| format!("{class}: {}", class.profile())).collect::<Vec<_>>().join(", ")
}

fn built_in_names() -> String {
    built_in().keys().cloned().collect::<Vec<_>>().join(", ")
}

/// `--profile none` / `IGNIS_PROFILE=none`: skip the profile layer
/// entirely, hardcoded defaults only. Not a name [`lookup`] resolves — a
/// config file's `profiles:` may not define one by this name either (nor one
/// called [`AUTO_PROFILE`]), since it would never be reachable (`gather`
/// checks for both first): `read_document` refuses it.
pub const NONE_PROFILE: &str = "none";

/// The RTX 5080 baseline settings. The historical name is retained so the
/// frozen coding-agent benchmark command still resolves the same values.
const BUILT_IN: &str = r#"{
    "rtx50-16gb": {
        "vram": {
            "qwen38": {
                "headroom_bytes": "1200M"
            }
        },
        "model": {
            "qwen38": {
                "max_context": 262144
            }
        },
        "vision": {
            "enabled": true,
            "max_tokens": 4096,
            "residency": "host"
        },
        "spec": {
            "qwen38": {
                "backend": "dflash2",
                "draft_tokens": 7,
                "decode_lanes": 2,
                "draft_head": "auto"
            }
        }
    }
}"#;

/// A **card class** (ADR 0063, ADR 0064, `GLOSSARY.md`): the cards one
/// built-in profile serves, named by their memory, and the smallest a
/// catalog entry loads on. Ordered by size, so `entry > card` reads "the
/// entry needs a larger card than this one".
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CardClass {
    Gb16,
    Gb32,
    Gb96,
}

/// The total memories each [`CardClass`] covers (spec 27b-16gb/04), as NVML
/// reports a card's total: below 20 GiB is the 16 GB class (a 16 GB card
/// reports ~15.9 GiB), 28-40 GiB the 32 GB one (the 5090 ~31.8 GiB), 88-104
/// GiB the 96 GB one (the RTX PRO 6000 ~95.6 GiB). Any other total is no
/// class: `--profile auto` refuses it, naming `--profile`.
pub const CARD_CLASS_16GB_BELOW_BYTES: u64 = 20 << 30;
pub const CARD_CLASS_32GB_BYTES: std::ops::RangeInclusive<u64> = (28 << 30)..=(40 << 30);
pub const CARD_CLASS_96GB_BYTES: std::ops::RangeInclusive<u64> = (88 << 30)..=(104 << 30);

impl CardClass {
    /// Every class, smallest first.
    pub const ALL: [CardClass; 3] = [Self::Gb16, Self::Gb32, Self::Gb96];

    /// The class a card whose total memory is `total_bytes` belongs to, by
    /// the boundaries above; `None` for a total between or past them.
    pub fn of_total(total_bytes: u64) -> Option<Self> {
        if total_bytes < CARD_CLASS_16GB_BELOW_BYTES {
            Some(Self::Gb16)
        } else if CARD_CLASS_32GB_BYTES.contains(&total_bytes) {
            Some(Self::Gb32)
        } else if CARD_CLASS_96GB_BYTES.contains(&total_bytes) {
            Some(Self::Gb96)
        } else {
            None
        }
    }

    /// The class's size in GB, as a catalog entry spells it (`card_class: 16`).
    pub fn gb(self) -> u32 {
        match self {
            Self::Gb16 => 16,
            Self::Gb32 => 32,
            Self::Gb96 => 96,
        }
    }

    /// The class a catalog entry's `card_class` names; `None` for any other
    /// number.
    pub fn from_gb(gb: u32) -> Option<Self> {
        Self::ALL.into_iter().find(|class| class.gb() == gb)
    }

    /// The built-in profile `--profile auto` picks for a card of this class.
    pub fn profile(self) -> &'static str {
        match self {
            Self::Gb16 => "rtx50-16gb",
            Self::Gb32 => "rtx5090",
            Self::Gb96 => "rtxpro6000",
        }
    }
}

impl std::fmt::Display for CardClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} GB", self.gb())
    }
}

/// Every built-in profile, by name.
pub fn built_in() -> BTreeMap<String, Layer> {
    let value = serde_json::from_str(&format!(r#"{{"profiles": {BUILT_IN}}}"#)).expect("the built-in profiles are JSON");
    read_document(&value, "the built-in profiles").expect("the built-in profiles name only declared fields").profiles
}

/// The profile called `name`: the config file's own first (an operator may
/// redefine a built-in one for their card), then the built-in ones.
pub fn lookup(name: &str, from_file: &BTreeMap<String, Layer>) -> Result<Layer, ConfigError> {
    if let Some(layer) = from_file.get(name) {
        return Ok(layer.clone());
    }
    let built_in = built_in();
    built_in.get(name).cloned().ok_or_else(|| {
        let mut known: Vec<&str> = built_in.keys().map(String::as_str).collect();
        known.extend(from_file.keys().map(String::as_str));
        known.sort_unstable();
        known.dedup();
        ConfigError(format!("no profile `{name}` (the profiles are {})", known.join(", ")))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::schema::field;

    #[test]
    fn the_only_built_in_profile_preserves_the_5080_baseline() {
        use super::super::source::{resolve_settings, Fit, Sources};
        use ignis_core::compute::ModelFamily;
        let profiles = built_in();
        assert_eq!(profiles.keys().map(String::as_str).collect::<Vec<_>>(), ["rtx50-16gb"]);
        let sources = Sources { profile: profiles["rtx50-16gb"].clone(), ..Sources::default() };
        let s = resolve_settings(&sources, Some(ModelFamily::Qwen38_27b), Fit::Start).unwrap().settings;
        assert_eq!(s.model.max_context, 262144);
        assert_eq!(s.spec.decode_lanes, Some(2));
        assert_eq!(s.spec.draft_tokens, Some(7));
        assert_eq!(s.vram.headroom_bytes, 1200 << 20);
        assert!(s.vision.enabled);
        assert_eq!(s.vision.max_tokens, 4096);
    }

    /// Spec 27b-16gb/04: the class boundaries by NVML's total, at the totals
    /// the cards report and either side of every boundary.
    #[test]
    fn a_cards_total_memory_names_its_class() {
        const GIB: f64 = (1u64 << 30) as f64;
        let class = |gib: f64| CardClass::of_total((gib * GIB) as u64);
        assert_eq!(class(15.9), Some(CardClass::Gb16), "a 16 GB card");
        assert_eq!(class(11.9), Some(CardClass::Gb16), "a 12 GB card is the 16 GB class's");
        assert_eq!(class(31.8), Some(CardClass::Gb32), "the 5090");
        assert_eq!(class(95.5), Some(CardClass::Gb96), "the RTX PRO 6000");
        assert_eq!(CardClass::of_total(CARD_CLASS_16GB_BELOW_BYTES - 1), Some(CardClass::Gb16));
        assert_eq!(CardClass::of_total(CARD_CLASS_16GB_BELOW_BYTES), None, "20 GiB: no class");
        assert_eq!(class(24.0), None, "a 24 GB card");
        assert_eq!(CardClass::of_total(*CARD_CLASS_32GB_BYTES.start() - 1), None);
        assert_eq!(CardClass::of_total(*CARD_CLASS_32GB_BYTES.start()), Some(CardClass::Gb32));
        assert_eq!(CardClass::of_total(*CARD_CLASS_32GB_BYTES.end()), Some(CardClass::Gb32));
        assert_eq!(CardClass::of_total(*CARD_CLASS_32GB_BYTES.end() + 1), None);
        assert_eq!(class(48.0), None, "a 48 GB card");
        assert_eq!(CardClass::of_total(*CARD_CLASS_96GB_BYTES.start()), Some(CardClass::Gb96));
        assert_eq!(CardClass::of_total(*CARD_CLASS_96GB_BYTES.end() + 1), None, "past 104 GiB");
        assert_eq!(class(141.0), None, "an H200");
    }

    #[test]
    fn a_card_class_is_named_by_its_gb_and_orders_by_size() {
        for class in CardClass::ALL {
            assert_eq!(CardClass::from_gb(class.gb()), Some(class));
        }
        assert_eq!(CardClass::from_gb(24), None);
        assert!(CardClass::Gb16 < CardClass::Gb32 && CardClass::Gb32 < CardClass::Gb96);
        assert_eq!(CardClass::Gb32.to_string(), "32 GB");
        assert_eq!(CardClass::Gb32.profile(), "rtx5090", "the 5090's class keeps its profile");
        assert_eq!(CardClass::Gb96.profile(), "rtxpro6000");
    }

    #[test]
    fn a_file_profile_shadows_a_built_in_one_and_an_unknown_name_lists_the_known() {
        let headroom = field("vram", "headroom_bytes").unwrap();
        let mut mine = Layer::default();
        mine.set(headroom, None, super::super::source::Candidate { raw: "2G".into(), spelling: "x".into() }).unwrap();
        let from_file = BTreeMap::from([("rtx50-16gb".to_owned(), mine.clone()), ("a4000".to_owned(), mine.clone())]);
        assert_eq!(lookup("rtx50-16gb", &from_file).unwrap(), mine);
        assert_eq!(lookup("rtx50-16gb", &BTreeMap::new()).unwrap(), built_in()["rtx50-16gb"]);
        let err = lookup("h100", &from_file).unwrap_err().0;
        assert!(err.contains("`h100`") && err.contains("a4000, rtx50-16gb"), "{err}");
    }
}
