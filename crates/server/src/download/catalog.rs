//! The catalog (ADR 0047, spec
//! `docs/specs/model-download/02-catalog-and-model-command.md`): every model
//! the server knows how to fetch, each pinned file by file.
//!
//! Two layers, with two trust standings. The **built-in catalog**
//! (`catalog.yaml`, compiled in with `include_str!`) holds the owner's
//! releases, and its pins are the trust anchor ADR 0033 made them. The
//! **operator catalog** is the file `download.catalog` names, in the same
//! format: the operator's own word about the operator's own files, never a
//! built-in entry's — an operator id equal to a built-in one is refused, so
//! no editable file can silently re-pin an official id.
//!
//! Both are read by [`parse`], which refuses a catalog whole: every refusal
//! names the file, the entry and the key, so the operator knows which line
//! to fix, and nothing downstream ever sees an entry that could write outside
//! its destination directory or fetch a file it cannot verify.
//!
//! An entry may state the smallest **card class** it loads on (every built-in
//! one does), and may list **aliases**: its family's default id, or another
//! entry's served id of its family, which it answers to on a card below that
//! id's own entry (ADR 0064, ADR 0074, spec 27b-16gb/04 §6). It may state its
//! **model family**, and must where an alias involves it and neither its id
//! nor its aliases name one: the header stays the authority, checked where an
//! alias resolves. [`Catalog::answering`] is the one class-aware step a
//! requested id goes through — at start, for a switch, a request, a download
//! and `GET /v1/models` — so a model the card cannot load is refused before
//! anything is torn down or fetched.

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use ignis_core::compute::{ModelFamily, ModelKind};
use serde_json::{Map, Value};

use crate::config::file::{Files, Format};
use crate::config::profile::CardClass;

/// The built-in catalog's text, as shipped.
const BUILT_IN_TEXT: &str = include_str!("catalog.yaml");

/// The built-in catalog, parsed once. A binary whose own catalog does not
/// parse is a build that must not ship: the test below parses it on every
/// `cargo test`.
pub static BUILT_IN_CATALOG: LazyLock<Catalog> = LazyLock::new(|| {
    let document = Format::Yaml.read(BUILT_IN_TEXT).expect("the built-in catalog is YAML");
    let entries = parse(&document, "built into this binary", CatalogLayer::BuiltIn).expect("the built-in catalog is valid");
    aliases_within_families(&entries).expect("the built-in catalog's aliases stay within their families");
    Catalog { entries }
});

/// The keys an entry takes; every one but `kind`, `card_class`, `family` and
/// `aliases` is required.
const ENTRY_KEYS: [&str; 9] = ["id", "repo", "revision", "artifact", "kind", "card_class", "family", "aliases", "files"];

/// The keys one of an entry's files takes.
const FILE_KEYS: [&str; 3] = ["name", "bytes", "sha256"];

/// Which catalog an entry came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CatalogLayer {
    /// Compiled into the binary: the owner's releases.
    BuiltIn,
    /// The file `download.catalog` names.
    Operator,
}

impl CatalogLayer {
    /// `built-in` or `operator`, as `model list` shows it.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BuiltIn => "built-in",
            Self::Operator => "operator",
        }
    }
}

/// One file an entry needs: the artifact, a sidecar, a companion container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogFile {
    /// Its name in the repo and on disk — a plain name, never a path, so the
    /// destination stays flat and nothing lands outside it.
    pub name: String,
    /// Its exact size.
    pub bytes: u64,
    /// Its SHA-256, lowercase hex.
    pub sha256: String,
}

/// One published model (`GLOSSARY.md`, **Catalog entry**): its served id,
/// where it is published, and every file it needs with its byte count and
/// digest pinned.
///
/// The pins are never read from the repo: the published `.sha256` sits in
/// the same trust domain as the artifact it describes, so it can attest
/// nothing about it (ADR 0033).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogEntry {
    /// The served id: what `--model-id` names and `GET /v1/models` reports.
    pub id: String,
    /// The repo, `<owner>/<name>`.
    pub repo: String,
    /// The commit, tag or branch the pins are of. The built-in entries name
    /// a commit, so a republish never breaks a binary already shipped.
    pub revision: String,
    /// The file the server loads, one of [`CatalogEntry::files`].
    pub artifact: String,
    /// What the entry is, when the catalog says it: `Some(ModelKind::Speech)`
    /// for a speech model (`kind: speech`; ADR 0067), never switched to,
    /// whose files are GGUFs with no sidecar; `Some(ModelKind::Diarizer)`
    /// for the diarizer (`kind: diarizer`; ADR 0069), one GGUF, also never
    /// switched to. `None` for the rest: a text or image model, told apart
    /// by its artifact's header.
    pub kind: Option<ModelKind>,
    /// The smallest card class it loads on (`card_class`, in GB; ADR 0064
    /// §1), when it states one. Optional in both layers, by one rule: an
    /// entry that states none is never refused by class, and the VRAM plan's
    /// refusal still guards it — so an operator catalog written before card
    /// classes still starts. Every built-in entry states one.
    pub card_class: Option<CardClass>,
    /// The model family it states (`family`, ADR 0074), by the family's own
    /// id, when it states one: required where an alias involves the entry
    /// and neither its id nor its aliases is a family's own id, and agreeing
    /// with them where they are. [`CatalogEntry::family()`] is the family the
    /// catalog tells, whichever says it.
    pub family: Option<ModelFamily>,
    /// The ids it also answers to (`aliases`, ADR 0064 §2): its family's
    /// default id, or another entry's served id of its family (ADR 0074), so
    /// that on a card class below that id's own entry a request naming it is
    /// served by this one. Usually none.
    pub aliases: Vec<String>,
    /// Every file, as the catalog lists them.
    pub files: Vec<CatalogFile>,
    /// Which catalog it came from.
    pub layer: CatalogLayer,
}

impl CatalogEntry {
    /// The model family the entry is of, as far as the catalog tells it
    /// (ADR 0074): the one it states, else the family whose own id it is, or
    /// whose own id it answers to as an alias — the catalog refuses an entry
    /// where these disagree. `None` for any other: no alias involves it, and
    /// the family stays the artifact's header's (ADR 0047).
    pub fn family(&self) -> Option<ModelFamily> {
        self.family
            .or_else(|| ModelFamily::of_model_id(&self.id))
            .or_else(|| self.aliases.iter().find_map(|alias| ModelFamily::of_model_id(alias)))
    }

    /// The model kind the catalog tells of the entry (GitHub #448): the
    /// `kind` it states (a speech model, the diarizer), else its family's
    /// ([`CatalogEntry::family()`]); `None` when it tells neither, which only
    /// the artifact's header can then say.
    pub fn model_kind(&self) -> Option<ModelKind> {
        self.kind.or_else(|| self.family().map(ModelFamily::kind))
    }

    /// Where the artifact lives under `dir` (flat: every file of an entry
    /// sits beside the others under the name the repo publishes it under).
    pub fn artifact_path(&self, dir: &Path) -> PathBuf {
        dir.join(&self.artifact)
    }

    /// Every file's bytes together: what a fetch of the whole entry spends.
    pub fn total_bytes(&self) -> u64 {
        self.files.iter().map(|file| file.bytes).sum()
    }

    /// [`CatalogEntry::total_bytes`] in GiB — for the question and the
    /// listing, never for a decision.
    pub fn gib(&self) -> f64 {
        (self.total_bytes() as f64) / (1024.0 * 1024.0 * 1024.0)
    }

    /// The files in the order they are fetched: ascending byte count, so a
    /// repo that cannot serve its small files fails in a second instead of
    /// after the large one.
    pub fn files_by_size(&self) -> Vec<&CatalogFile> {
        let mut files: Vec<&CatalogFile> = self.files.iter().collect();
        files.sort_by(|a, b| a.bytes.cmp(&b.bytes).then_with(|| a.name.cmp(&b.name)));
        files
    }
}

/// The models a server knows how to fetch: the built-in entries, then the
/// operator's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Catalog {
    entries: Vec<CatalogEntry>,
}

impl Catalog {
    /// The built-in catalog alone.
    pub fn built_in() -> &'static Catalog {
        &BUILT_IN_CATALOG
    }

    /// A catalog of exactly `entries`: for a server built in process, which
    /// reads no catalog file (the speech slot's tests, whose models are
    /// fixtures).
    pub fn from_entries(entries: Vec<CatalogEntry>) -> Catalog {
        Catalog { entries }
    }

    /// Every entry, the built-in ones first.
    pub fn entries(&self) -> &[CatalogEntry] {
        &self.entries
    }

    /// The entry for `id`, matched case-insensitively: a served id is a
    /// name, not a checksum.
    pub fn entry(&self, id: &str) -> Option<&CatalogEntry> {
        self.entries.iter().find(|entry| entry.id.eq_ignore_ascii_case(id.trim()))
            .or_else(|| self.entries.iter().find(|entry| entry.aliases.iter().any(|alias| alias.eq_ignore_ascii_case(id.trim()))))
    }

    /// Every entry's id, in order — what an unknown id is answered with.
    pub fn ids(&self) -> Vec<&str> {
        self.entries.iter().map(|entry| entry.id.as_str()).collect()
    }

    /// The class rule (ADR 0064 §1, §3), the one place it is written: the
    /// entry `id` itself needs a larger card than one of class `card` — its
    /// refusal, naming both classes. `None` when it fits, states no class,
    /// no class is known (`None`: no total read, or one of no class — the
    /// VRAM plan's refusal still guards), or no entry has the id: an
    /// operator's own model is not the catalog's to judge.
    ///
    /// Ids are matched exactly, as a request's `model`, the switch's known
    /// models and the engine match them — not case-insensitively as
    /// [`Catalog::entry`] does for the `model` command line, whose callers
    /// take the entry's own id first.
    pub fn above_class(&self, id: &str, card: Option<CardClass>) -> Option<ClassRefusal> {
        let card = card?;
        let entry = self.entries.iter().find(|entry| entry.id == id)?;
        let needs = entry.card_class.filter(|needs| *needs > card)?;
        Some(ClassRefusal { model: entry.id.clone(), needs, card, other_family: None })
    }

    /// The class-aware step (ADR 0064 §2-3, spec 27b-16gb/04 §6) every
    /// adapter takes — the start, a switch, a request, a download, the model
    /// lists: which entry answers to `requested` on a card of class `card`
    /// in its place. `Ok(None)` when `requested` stands as it is — its own
    /// entry is not [`Catalog::above_class`]; `Ok(Some(entry))` when its own
    /// entry is, and `entry`, of its family, lists it as an alias and fits
    /// the card (or states no class); refused, naming both classes, when no
    /// entry that answers to it fits. Matched exactly, as
    /// [`Catalog::above_class`] matches. The family is the one the catalog
    /// tells of `requested`'s entry ([`CatalogEntry::family()`]): its id's,
    /// or the one it states (ADR 0074) — an entry of no family the catalog
    /// tells has none to answer to it.
    pub fn answering(&self, requested: &str, card: Option<CardClass>) -> Result<Option<&CatalogEntry>, ClassRefusal> {
        if !self.entries.iter().any(|entry| entry.id == requested) {
            if let Some(entry) = self.entries.iter().find(|entry| entry.aliases.iter().any(|alias| alias == requested)) {
                if let Some(refusal) = self.above_class(&entry.id, card) { return Err(refusal); }
                return Ok(Some(entry));
            }
        }
        let Some(refusal) = self.above_class(requested, card) else {
            return Ok(None);
        };
        let family = self.family_of(requested);
        let answering = self.entries.iter().find(|entry| {
            entry.card_class.is_none_or(|needs| needs <= refusal.card)
                && family.is_some()
                && entry.family() == family
                && entry.aliases.iter().any(|alias| alias == requested)
        });
        answering.map(Some).ok_or(refusal)
    }

    /// The family the catalog tells of the entry whose id is exactly `id`
    /// ([`CatalogEntry::family()`]); `None` for none, or no such entry.
    fn family_of(&self, id: &str) -> Option<ModelFamily> {
        self.entry(id).and_then(CatalogEntry::family)
    }

    /// ADR 0064 §2 on the artifact itself, read once it is on disk: `entry`,
    /// which answers to `requested` on a card of class `card`
    /// ([`Catalog::answering`]), holds an artifact whose header names
    /// `header` (ADR 0047: the header is the authority, the catalog's family
    /// a claim it checks). An alias only ever resolves within one model
    /// family, so that must be `requested`'s family as the catalog tells it
    /// — its id's, or the one its entry states (ADR 0074); else the alias
    /// does not resolve, refused naming both families — a header that names
    /// no family is refused too, the safe side. `Ok` when no alias is in
    /// play: `requested`'s own entry is not above the card's class.
    pub fn alias_holds(&self, requested: &str, entry: &CatalogEntry, header: Option<ModelFamily>, card: Option<CardClass>) -> Result<(), ClassRefusal> {
        let mut refusal = match self.above_class(requested, card) {
            Some(refusal) => refusal,
            None if entry.id != requested && entry.aliases.iter().any(|alias| alias == requested) => ClassRefusal {
                model: requested.to_owned(), needs: entry.card_class.unwrap_or(CardClass::Gb16),
                card: card.unwrap_or(CardClass::Gb16), other_family: None,
            },
            None => return Ok(()),
        };
        let alias_family = self.family_of(requested);
        if header.is_some() && header == alias_family {
            return Ok(());
        }
        refusal.other_family = Some(OtherFamily { entry: entry.id.clone(), artifact: header, alias_family });
        Err(refusal)
    }
}

/// A catalog entry the card cannot load (ADR 0064 §3): refused before
/// anything is torn down or fetched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassRefusal {
    /// The entry's id: the one requested.
    pub model: String,
    /// The entry's `card_class`.
    pub needs: CardClass,
    /// The card's class.
    pub card: CardClass,
    /// The entry that lists `model` as an alias and fits this card, when one
    /// does but its artifact is of another family than `model`'s
    /// ([`Catalog::alias_holds`]): it does not answer to the alias.
    pub other_family: Option<OtherFamily>,
}

/// An entry that would answer to an alias but holds an artifact of another
/// family than the alias's (ADR 0064 §2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OtherFamily {
    /// The entry's id.
    pub entry: String,
    /// The family its artifact's header names; `None` for none.
    pub artifact: Option<ModelFamily>,
    /// The alias's family, as the catalog tells it (ADR 0074): the family
    /// whose own id the alias is, or the one its entry states.
    pub alias_family: Option<ModelFamily>,
}

impl std::fmt::Display for ClassRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} needs a {} card or larger (its catalog entry's card_class), and this card is of the {} class", self.model, self.needs, self.card)?;
        if let Some(other) = &self.other_family {
            let alias_family = other.alias_family.map_or("no model family", ModelFamily::name);
            let artifact = other.artifact.map_or("no model family", ModelFamily::name);
            write!(
                f,
                "; {} lists it as an alias, but its artifact is {artifact} and {} is {alias_family}: an alias resolves only within one model family",
                other.entry, self.model
            )?;
        }
        Ok(())
    }
}

impl std::error::Error for ClassRefusal {}

/// A catalog that cannot be used: unreadable, malformed, or naming an entry
/// it may not. Refuses the start, and every `model` subcommand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogError(pub String);

impl std::fmt::Display for CatalogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for CatalogError {}

/// The catalog a server runs with: the built-in one, and the operator's file
/// over it when `path` (`download.catalog`, already resolved against the
/// config file that named it) names one. A path that cannot be read is
/// refused, not skipped: an operator who named a catalog meant it.
pub fn load(files: &dyn Files, path: Option<&Path>) -> Result<Catalog, CatalogError> {
    let Some(path) = path else {
        return Ok(Catalog::built_in().clone());
    };
    let text = files
        .read(path)
        .map_err(|err| CatalogError(format!("the catalog {} (`download.catalog`) cannot be read: {err}", path.display())))?;
    parse_operator(&text, Format::of_file(path), &path.display().to_string())
}

/// An operator catalog's text, read in `format` and put after the built-in
/// entries. `origin` (the file's path) is named in every refusal.
///
/// An operator entry may answer to a built-in entry's id as an alias (ADR
/// 0064 §2: an operator's own 16 GB build of the 27B answering to
/// `qwen3.8-27b`). That is not the silent re-pin ADR 0047 refuses: the
/// built-in entry keeps its id and its pins and serves it wherever it fits,
/// the alias is consulted only on a card below its class, and the served id
/// a response reports is the operator entry's, so a client can tell which
/// model answered. An alias a built-in entry already claims is refused, as
/// within one file.
pub fn parse_operator(text: &str, format: Format, origin: &str) -> Result<Catalog, CatalogError> {
    parse_operator_over(Catalog::built_in(), text, format, origin)
}

/// [`parse_operator`] over `built_in` in the built-in catalog's place: the
/// seam a test puts a built-in entry with an alias behind, which the shipped
/// catalog does not hold yet.
fn parse_operator_over(built_in: &Catalog, text: &str, format: Format, origin: &str) -> Result<Catalog, CatalogError> {
    let document = format
        .read(text)
        .map_err(|err| CatalogError(format!("the catalog {origin} is not valid: {err}")))?;
    let operator = parse(&document, origin, CatalogLayer::Operator)?;
    let mut entries = built_in.entries.clone();
    for entry in operator {
        if let Some(official) = built_in.entry(&entry.id) {
            return Err(CatalogError(format!(
                "the catalog {origin}: entry `{}` reuses the built-in id `{}` — an operator entry never re-pins an official model (a mirror of it is `download.endpoint`, a variant of it a new id)",
                entry.id, official.id
            )));
        }
        entries.push(entry);
    }
    claimed_once(&entries).map_err(|reason| CatalogError(format!("the catalog {origin}: {reason}")))?;
    aliases_within_families(&entries).map_err(|reason| CatalogError(format!("the catalog {origin}: {reason}")))?;
    Ok(Catalog { entries })
}

/// A catalog document's entries, every one validated: the one reader both
/// layers go through, so the built-in catalog is held to the rules an
/// operator's is.
pub fn parse(document: &Value, origin: &str, layer: CatalogLayer) -> Result<Vec<CatalogEntry>, CatalogError> {
    let refuse = |reason: String| CatalogError(format!("the catalog {origin}: {reason}"));
    let Value::Object(top) = document else {
        return Err(refuse("must be a map holding a `models` list".to_owned()));
    };
    if let Some(key) = top.keys().find(|key| *key != "models") {
        return Err(refuse(format!("`{key}` is not a catalog key (the one key is `models`)")));
    }
    let Some(Value::Array(models)) = top.get("models") else {
        return Err(refuse("`models` must be a list of entries".to_owned()));
    };
    let mut entries: Vec<CatalogEntry> = Vec::new();
    for (index, value) in models.iter().enumerate() {
        let named = match value.get("id").and_then(Value::as_str).map(str::trim).filter(|id| !id.is_empty()) {
            Some(id) => format!("entry {} (`{id}`)", index + 1),
            None => format!("entry {}", index + 1),
        };
        let entry = read_entry(value, layer).map_err(|reason| refuse(format!("{named}: {reason}")))?;
        if let Some(earlier) = entries.iter().find(|earlier| earlier.id.eq_ignore_ascii_case(&entry.id)) {
            return Err(refuse(format!(
                "{named}: the id is already `{}`'s, an earlier entry (ids are compared case-insensitively)",
                earlier.id
            )));
        }
        entries.push(entry);
    }
    claimed_once(&entries).map_err(refuse)?;
    Ok(entries)
}

/// Every alias claimed by one entry at most (ADR 0064 §2): an id answers for
/// one entry, or which one served it would be the catalog's order. Checked
/// over one file's entries, and again over the merged catalog once an
/// operator's join the built-in ones. That an alias is another entry's id
/// is the point of it, not a collision.
fn claimed_once(entries: &[CatalogEntry]) -> Result<(), String> {
    for (index, entry) in entries.iter().enumerate() {
        for alias in &entry.aliases {
            if let Some(earlier) = entries[..index].iter().find(|earlier| earlier.aliases.contains(alias)) {
                return Err(format!(
                    "entry `{}`: alias `{alias}` is already `{}`'s — an id answers for one entry",
                    entry.id, earlier.id
                ));
            }
        }
    }
    Ok(())
}

/// Every alias within its entry's model family (ADR 0064 §2, ADR 0074),
/// over a whole catalog — the built-in one, and the merged one once an
/// operator's entries join it, since an operator's alias may be a built-in
/// entry's id. An alias that is an entry's served id, matched exactly as a
/// request's `model` is, names an entry of a family the catalog tells, and
/// of the aliasing entry's own: so an entry another entry's alias names
/// states `family` when its id and aliases name none. Any other alias is a
/// family's own id, which [`aliases`] held to the entry's family already.
fn aliases_within_families(entries: &[CatalogEntry]) -> Result<(), String> {
    for entry in entries {
        for alias in &entry.aliases {
            let Some(named) = entries.iter().find(|named| named.id == *alias) else {
                if ModelFamily::of_model_id(alias).is_some() {
                    continue;
                }
                return Err(format!(
                    "entry `{}`: alias `{alias}` is no model family's own id ({}) and no entry's served id (ids match exactly): an alias answers for an id on a smaller card",
                    entry.id,
                    family_ids()
                ));
            };
            match (named.family(), entry.family()) {
                (None, _) => {
                    return Err(format!(
                        "entry `{}`: alias `{alias}` names an entry whose family nothing states — its id and aliases name none, so `{}` must state `family` ({}): an alias resolves only within one model family",
                        entry.id,
                        named.id,
                        family_ids()
                    ))
                }
                (Some(theirs), Some(own)) if theirs != own => {
                    return Err(format!(
                        "entry `{}`: alias `{alias}` names an entry of {} and the entry is of {}: an alias resolves only within one model family",
                        entry.id,
                        theirs.name(),
                        own.name()
                    ))
                }
                _ => {}
            }
        }
    }
    Ok(())
}

/// Every model family's own id, as a refusal lists them.
fn family_ids() -> String {
    ModelFamily::ALL.iter().map(|family| family.model_id()).collect::<Vec<_>>().join(", ")
}

/// One entry, or why it is refused (the caller names the entry).
fn read_entry(value: &Value, layer: CatalogLayer) -> Result<CatalogEntry, String> {
    let Value::Object(map) = value else {
        return Err(format!("must be a map of {}", ENTRY_KEYS.join(", ")));
    };
    known_keys(map, &ENTRY_KEYS, "an entry")?;
    let id = text(map, "id")?;
    let repo = text(map, "repo")?;
    let plain_part = |part: &str| plain_name(part) && !part.contains(char::is_whitespace);
    if !repo.split_once('/').is_some_and(|(owner, name)| plain_part(owner) && plain_part(name)) {
        return Err(format!("`repo` must be `<owner>/<name>`, got `{repo}`"));
    }
    let revision = text(map, "revision")?;
    let artifact = text(map, "artifact")?;
    let kind = kind(map)?;
    let card_class = card_class(map)?;
    let family = family(map, &id)?;
    let aliases = aliases(map, &id, family)?;
    let Some(Value::Array(listed)) = map.get("files") else {
        return Err("`files` must be a list of `{ name, bytes, sha256 }`".to_owned());
    };
    let mut files: Vec<CatalogFile> = Vec::new();
    for (index, value) in listed.iter().enumerate() {
        let named = match value.get("name").and_then(Value::as_str) {
            Some(name) => format!("file {} (`{name}`)", index + 1),
            None => format!("file {}", index + 1),
        };
        let file = read_file(value).map_err(|reason| format!("{named}: {reason}"))?;
        if files.iter().any(|earlier| earlier.name == file.name) {
            return Err(format!("{named}: listed twice"));
        }
        files.push(file);
    }
    if !files.iter().any(|file| file.name == artifact) {
        return Err(format!("`artifact` `{artifact}` is not among its `files`"));
    }
    let sidecars: Vec<String> = crate::loader::SIDECAR_SUFFIXES.iter().map(|suffix| format!("{artifact}{suffix}")).collect();
    if !files.iter().any(|file| sidecars.contains(&file.name)) {
        return Err(format!("no file is the artifact's sidecar ({}): the loader refuses a load without one", sidecars.join(" or ")));
    }
    Ok(CatalogEntry { id, repo, revision, artifact, kind, card_class, family, aliases, files, layer })
}

/// `kind`, when stated: `speech` or `diarizer`, the kinds a catalog says. A
/// text or an image model is told apart by its artifact's header (ADR 0047),
/// so stating either here would be a second word that could disagree with
/// the file.
fn kind(map: &Map<String, Value>) -> Result<Option<ModelKind>, String> {
    match map.get("kind") {
        None => Ok(None),
        Some(other) => Err(format!("`kind` is unsupported in this fork: only Qwen3.8-27B artifacts are served, got {other}")),
    }
}

/// `card_class`, when stated: the GB of the smallest card class the entry
/// loads on, one of [`CardClass::ALL`]'s — `CardClass::from_gb` reads it, so
/// any other number refuses the catalog. Left out, the entry states none.
fn card_class(map: &Map<String, Value>) -> Result<Option<CardClass>, String> {
    let classes = CardClass::ALL.map(|class| class.gb().to_string()).join(", ");
    let refuse = |got: &dyn std::fmt::Display| format!("`card_class` must be {classes} (the GB of the smallest card it loads on), got {got}");
    match map.get("card_class") {
        Some(Value::Number(number)) => {
            number.as_u64().and_then(|gb| u32::try_from(gb).ok()).and_then(CardClass::from_gb).map(Some).ok_or_else(|| refuse(number))
        }
        Some(other) => Err(refuse(other)),
        None => Ok(None),
    }
}

/// `family`, when stated: a model family's own id ([`ModelFamily::ALL`]),
/// matched exactly, and the family the entry's own id is, when it is one's
/// (ADR 0074). Its aliases are held to it by [`aliases`].
fn family(map: &Map<String, Value>, id: &str) -> Result<Option<ModelFamily>, String> {
    let Some(value) = map.get("family") else {
        return Ok(None);
    };
    let Some(stated) = value.as_str().and_then(|stated| ModelFamily::of_model_id(stated.trim())) else {
        return Err(format!("`family` must be a model family's own id ({}), got {value}", family_ids()));
    };
    if let Some(own) = ModelFamily::of_model_id(id).filter(|own| *own != stated) {
        return Err(format!(
            "`family` is {} and the entry's own id `{id}` is {}'s: a stated family agrees with what the id says",
            stated.name(),
            own.name()
        ));
    }
    Ok(Some(stated))
}

/// `aliases`, when listed: none the entry's own id, none twice. One that is
/// a model family's own id is of the family the entry's own id names, when
/// it names one, of the family it states (`stated`), and of the family its
/// other such aliases are. One that is not names another entry's served id,
/// held to the entry's family over the whole catalog
/// ([`aliases_within_families`]) — so an entry listing one, whose id and
/// aliases name no family, states `family` (ADR 0074). That is what keeps an
/// alias within one family.
fn aliases(map: &Map<String, Value>, id: &str, stated: Option<ModelFamily>) -> Result<Vec<String>, String> {
    let listed = match map.get("aliases") {
        None => return Ok(Vec::new()),
        Some(Value::Array(listed)) => listed,
        Some(other) => return Err(format!("`aliases` must be a list of served ids, got {other}")),
    };
    let mut aliases: Vec<String> = Vec::new();
    for value in listed {
        let Some(alias) = value.as_str().map(str::trim).filter(|alias| !alias.is_empty()) else {
            return Err(format!("`aliases` must be a list of served ids, got {value}"));
        };
        if alias.eq_ignore_ascii_case(id) {
            return Err(format!("`aliases` lists the entry's own id `{alias}`"));
        }
        if aliases.iter().any(|earlier| earlier == alias) {
            return Err(format!("alias `{alias}` is listed twice"));
        }
        if let Some(family) = ModelFamily::of_model_id(alias) {
            if let Some(own) = ModelFamily::of_model_id(id).filter(|own| *own != family) {
                return Err(format!(
                    "alias `{alias}` is {}'s own id and the entry's own id `{id}` is {}'s: an alias resolves only within one model family",
                    family.name(),
                    own.name()
                ));
            }
            if let Some(stated) = stated.filter(|stated| *stated != family) {
                return Err(format!(
                    "alias `{alias}` is {}'s own id and the entry states `family` {}: an alias resolves only within one model family",
                    family.name(),
                    stated.name()
                ));
            }
            if let Some(first) = aliases.iter().find(|earlier| ModelFamily::of_model_id(earlier).is_some_and(|earlier| earlier != family)) {
                return Err(format!("aliases `{first}` and `{alias}` name two model families: an alias resolves only within one"));
            }
        }
        aliases.push(alias.to_owned());
    }
    let named = ModelFamily::of_model_id(id).is_some() || aliases.iter().any(|alias| ModelFamily::of_model_id(alias).is_some());
    if !aliases.is_empty() && stated.is_none() && !named {
        return Err(format!(
            "it lists aliases, and neither its id nor its aliases is a model family's own id, so it must state `family` ({}), the family it is of (ADR 0074)",
            family_ids()
        ));
    }
    Ok(aliases)
}

/// One of an entry's files, or why it is refused.
fn read_file(value: &Value) -> Result<CatalogFile, String> {
    let Value::Object(map) = value else {
        return Err(format!("must be a map of {}", FILE_KEYS.join(", ")));
    };
    known_keys(map, &FILE_KEYS, "a file")?;
    let name = text(map, "name")?;
    if !plain_name(&name) {
        return Err(format!(
            "`{name}` is not a plain file name (letters, digits, `.`, `_` and `-`, not starting or ending with `.`, not a Windows device name): a catalog never writes outside the destination directory"
        ));
    }
    let bytes = match map.get("bytes") {
        Some(Value::Number(number)) => number.as_u64().ok_or_else(|| format!("`bytes` must be a whole number of bytes, got {number}"))?,
        Some(other) => return Err(format!("`bytes` must be a whole number of bytes, got {other}")),
        None => return Err("has no `bytes`".to_owned()),
    };
    if bytes == 0 {
        return Err("`bytes` is 0: an empty file pins nothing".to_owned());
    }
    let sha256 = text(map, "sha256")?;
    if sha256.len() != 64 || !sha256.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err(format!("`sha256` must be 64 lowercase hex digits, got `{sha256}`"));
    }
    Ok(CatalogFile { name, bytes, sha256 })
}

/// Refuse a key `map` should not hold, naming it and the ones it may.
fn known_keys(map: &Map<String, Value>, keys: &[&str], what: &str) -> Result<(), String> {
    match map.keys().find(|key| !keys.contains(&key.as_str())) {
        Some(key) => Err(format!("`{key}` is not a key of {what} (the keys are {})", keys.join(", "))),
        None => Ok(()),
    }
}

/// `map[key]` as non-empty text, trimmed.
fn text(map: &Map<String, Value>, key: &str) -> Result<String, String> {
    match map.get(key) {
        Some(Value::String(text)) if !text.trim().is_empty() => Ok(text.trim().to_owned()),
        Some(Value::String(_)) | Some(Value::Null) | None => Err(format!("`{key}` is empty or missing")),
        Some(other) => Err(format!("`{key}` must be text, got {other}")),
    }
}

/// A name that stays inside the directory it is joined to on every host. A
/// security predicate lists the safe side, not the unsafe one: only letters,
/// digits, `.`, `_` and `-` (so no separator of either kind, no drive, no
/// control character, no space), neither starting nor ending with `.`
/// (`.`, `..`, and the trailing dot Windows silently drops), and not a
/// Windows device name (`CON`, `NUL`, `COM1`...), which Windows opens as the
/// device whatever follows its first `.`.
fn plain_name(name: &str) -> bool {
    const DEVICES: [&str; 22] = [
        "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5",
        "LPT6", "LPT7", "LPT8", "LPT9",
    ];
    let stem = name.split('.').next().unwrap_or(name);
    !name.is_empty()
        && name.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
        && !name.starts_with('.')
        && !name.ends_with('.')
        && !DEVICES.iter().any(|device| stem.eq_ignore_ascii_case(device))
}

/// Test fixtures other modules' tests share.
#[cfg(test)]
pub(crate) mod testing {
    /// The 16 GB 27B's entry as an operator catalog would state it: its own
    /// served id, the 16 GB class, and the 27B's default id as its alias.
    /// The built-in catalog holds the real one (GitHub #338), so this one is
    /// parsed over [`thirty_two`]: the input of the parsing tests.
    pub(crate) const SIXTEEN: &str = include_str!("../../tests/fixtures/catalog/sixteen.yaml");

    /// The built-in catalog, which holds the 16 GB 27B (GitHub #338) and the
    /// uncensored variant's 16 GB build (GitHub #423).
    pub(crate) fn built_in() -> super::Catalog {
        super::Catalog::built_in().clone()
    }

    /// The built-in catalog without its 16 GB entries: a catalog in which no
    /// entry answers to the 27B's default id, nor to the uncensored
    /// variant's, on a 16 GB card.
    pub(crate) fn thirty_two() -> super::Catalog {
        let entries = super::Catalog::built_in().entries.iter().filter(|entry| entry.card_class != Some(super::CardClass::Gb16));
        super::Catalog { entries: entries.cloned().collect() }
    }

    /// [`SIXTEEN`] as an operator's catalog over [`thirty_two`].
    pub(crate) fn sixteen_over_thirty_two(text: &str) -> Result<super::Catalog, super::CatalogError> {
        super::parse_operator_over(&thirty_two(), text, crate::config::file::Format::Yaml, "acme.catalog.yaml")
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{sixteen_over_thirty_two, thirty_two, SIXTEEN};
    use super::*;

    /// A minimal valid operator entry, YAML, for the refusal table to break
    /// one thing at a time.
    const OPERATOR: &str = "models:
  - id: acme-qwen3.8-27b-ft
    repo: acme/qwen-ft-ninfer
    revision: v1
    artifact: acme_ft.ninfer
    card_class: 32
    files:
      - { name: acme_ft.ninfer.graft.json, bytes: 48211, sha256: 9c1e000000000000000000000000000000000000000000000000000000000001 }
      - { name: acme_ft.ninfer, bytes: 19406942468, sha256: abb1000000000000000000000000000000000000000000000000000000000002 }
";

    fn operator(text: &str) -> Result<Catalog, CatalogError> {
        parse_operator(text, Format::Yaml, "acme.catalog.yaml")
    }

    /// AC 2: an operator catalog adds its entries after the built-in ones,
    /// read as YAML or as JSON.
    #[test]
    fn an_operator_catalog_adds_its_entries_in_either_format() {
        let from_yaml = operator(OPERATOR).expect("valid");
        let json = serde_json::to_string(&Format::Yaml.read(OPERATOR).unwrap()).unwrap();
        let from_json = parse_operator(&json, Format::Json, "acme.catalog.json").expect("valid");
        assert_eq!(from_yaml, from_json);
        assert_eq!(from_yaml.ids(), ["qwen3.8-27b-16g", "acme-qwen3.8-27b-ft"]);
        let acme = from_yaml.entry("acme-qwen3.8-27b-ft").unwrap();
        assert_eq!(acme.layer, CatalogLayer::Operator);
        assert_eq!(acme.revision, "v1");
        assert_eq!(acme.artifact_path(Path::new("models")), Path::new("models").join("acme_ft.ninfer"));
        assert_eq!(acme.total_bytes(), 48_211 + 19_406_942_468);
    }

    /// AC 2: every refusal *The catalog file* lists, each naming the entry
    /// or the key it is about — the built-in-id collision and the non-plain
    /// file name among them.
    #[test]
    fn each_refusal_names_the_offending_entry_or_key() {
        let entry = "acme-qwen3.8-27b-ft";
        for (from, to, says) in [
            ("    revision: v1\n", "    revision: v1\n    mirror: x\n", vec!["`acme-qwen3.8-27b-ft`", "`mirror`"]),
            ("models:\n", "endpoint: x\nmodels:\n", vec!["`endpoint`"]),
            ("bytes: 48211,", "bytes: 48211, size: 1,", vec!["`size`", "acme_ft.ninfer.graft.json"]),
            ("id: acme-qwen3.8-27b-ft", "id: \"\"", vec!["entry 1", "`id`"]),
            ("id: acme-qwen3.8-27b-ft", "id: QWEN3.8-27B-16G", vec!["`QWEN3.8-27B-16G`", "built-in id `qwen3.8-27b-16g`"]),
            ("repo: acme/qwen-ft-ninfer", "repo: acme-qwen-ft-ninfer", vec![entry, "`repo`", "<owner>/<name>"]),
            ("repo: acme/qwen-ft-ninfer", "repo: acme/qwen/ft", vec![entry, "`repo`"]),
            ("repo: acme/qwen-ft-ninfer", "repo: /qwen", vec![entry, "`repo`"]),
            ("revision: v1", "revision: \"\"", vec![entry, "`revision`"]),
            ("artifact: acme_ft.ninfer", "artifact: acme_other.ninfer", vec![entry, "`acme_other.ninfer`", "not among"]),
            ("acme_ft.ninfer.graft.json", "acme_ft.ninfer.notes.json", vec![entry, "sidecar", "acme_ft.ninfer.graft.json"]),
            ("sha256: 9c1e0", "sha256: 9C1E0", vec![entry, "`sha256`", "lowercase"]),
            ("sha256: abb1000000000000000000000000000000000000000000000000000000000002", "sha256: abb1", vec![entry, "`sha256`", "64"]),
            ("bytes: 48211", "bytes: 0", vec![entry, "`bytes` is 0"]),
            ("bytes: 48211", "bytes: lots", vec![entry, "`bytes`"]),
            // ADR 0064 §1: a card class stated is one of the classes there
            // are.
            ("card_class: 32", "card_class: 24", vec![entry, "`card_class`", "16, 32, 96", "got 24"]),
            ("card_class: 32", "card_class: \"32\"", vec![entry, "`card_class`", "16, 32, 96"]),
            ("card_class: 32", "card_class: -16", vec![entry, "`card_class`"]),
            // ADR 0064 §2: an alias is of one family, never the entry's own
            // id.
            ("card_class: 32", "card_class: 32\n    aliases: qwen3.8-27b", vec![entry, "`aliases` must be a list"]),
            ("card_class: 32", "card_class: 32\n    aliases: [acme-qwen3.8-27b-ft]", vec![entry, "own id"]),
            ("card_class: 32", "card_class: 32\n    aliases: [qwen3.8-27b, qwen3.8-flash-next]", vec![entry, "`qwen3.8-27b`", "`qwen3.8-flash-next`", "two model families"]),
            ("card_class: 32", "card_class: 32\n    aliases: [qwen3.8-27b, qwen3.8-27b]", vec![entry, "`qwen3.8-27b`", "twice"]),
        ] {
            assert!(OPERATOR.contains(from), "{from}");
            let text = OPERATOR.replacen(from, to, 1);
            let err = operator(&text).expect_err(&text).0;
            assert!(err.contains("acme.catalog.yaml"), "{err}");
            for said in says {
                assert!(err.contains(said), "{to:?}: `{said}` missing from: {err}");
            }
        }
    }

    #[test]
    fn a_duplicate_id_is_refused_case_insensitively_within_the_file() {
        let body = OPERATOR.strip_prefix("models:\n").unwrap();
        let twice = format!("{OPERATOR}{}", body.replacen("acme-qwen3.8-27b-ft", "ACME-Qwen3.8-27B-FT", 1));
        let err = operator(&twice).unwrap_err().0;
        assert!(err.contains("entry 2 (`ACME-Qwen3.8-27B-FT`)") && err.contains("`acme-qwen3.8-27b-ft`"), "{err}");
    }

    /// ADR 0064 §1 as built: `card_class` is optional, by one rule — an
    /// entry that states none is never refused by class, as its own id or
    /// answering to an alias; the VRAM plan's refusal still guards it. So an
    /// operator catalog written before card classes still starts.
    #[test]
    fn an_entry_that_states_no_card_class_is_never_refused_by_class() {
        use CardClass::*;
        let catalog = operator(&OPERATOR.replacen("    card_class: 32\n", "", 1)).expect("a catalog from before card classes");
        let acme = catalog.entry("acme-qwen3.8-27b-ft").unwrap();
        assert_eq!(acme.card_class, None);
        for card in [Some(Gb16), Some(Gb32), None] {
            assert_eq!(catalog.above_class(&acme.id, card), None, "{card:?}");
            assert_eq!(resolved(&catalog, &acme.id, card), Ok(acme.id.as_str()), "{card:?}");
        }
        let classless = sixteen_over_thirty_two(&SIXTEEN.replacen("    card_class: 16\n", "", 1)).unwrap();
        assert_eq!(resolved(&classless, "qwen3.8-27b", Some(Gb16)), Ok("qwen3.8-27b-16g"), "it answers to its alias on any card");
    }

    /// ADR 0064 §2: an alias resolves only within one model family — an
    /// entry whose own id is a family's may not answer to another's.
    #[test]
    fn an_alias_of_another_family_than_the_entrys_own_id_is_refused() {
        let text = OPERATOR.replacen("acme-qwen3.8-27b-ft", "qwen3.8-flash-next", 1).replacen("card_class: 32", "card_class: 32\n    aliases: [qwen3.8-27b]", 1);
        let err = parse(&Format::Yaml.read(&text).unwrap(), "test", CatalogLayer::BuiltIn).unwrap_err().0;
        assert!(err.contains("`qwen3.8-flash-next`") && err.contains("`qwen3.8-27b`") && err.contains("one model family"), "{err}");
    }

    /// ADR 0064 §2: an id answers for one entry, or which one served it would
    /// be the catalog's order — refused within a file.
    #[test]
    fn an_alias_two_entries_claim_is_refused() {
        let (_, body) = SIXTEEN.split_once("models:\n").unwrap();
        let first = body.split("\n  - ").next().unwrap();
        let twice = format!("{SIXTEEN}\n{}\n", first.replacen("id: qwen3.8-27b-16g", "id: acme-27b-16g", 1));
        let err = sixteen_over_thirty_two(&twice).unwrap_err().0;
        assert!(
            err.contains("acme.catalog.yaml") && err.contains("`acme-27b-16g`") && err.contains("`qwen3.8-27b-16g`") && err.contains("alias `qwen3.8-27b`"),
            "{err}"
        );
    }

    /// ADR 0064 §2 across the two layers: an operator entry claiming an alias
    /// a built-in entry claims is refused by the check over the merged
    /// catalog, each file being valid on its own; an operator entry reusing
    /// the built-in 16 GB 27B's id is refused as any built-in id is.
    #[test]
    fn an_alias_a_built_in_entry_claims_is_refused_to_an_operator_entry() {
        let claiming = SIXTEEN.replacen("id: qwen3.8-27b-16g", "id: acme-27b-16g", 1);
        let err = operator(&claiming).unwrap_err().0;
        assert!(err.contains("acme.catalog.yaml") && err.contains("entry `acme-27b-16g`: alias `qwen3.8-27b` is already `qwen3.8-27b-16g`'s"), "{err}");
        assert!(operator(SIXTEEN).unwrap_err().0.contains("reuses the built-in id `qwen3.8-27b-16g`"));
        let joined = operator(OPERATOR).expect("an entry claiming no alias joins");
        assert_eq!(joined.ids().last(), Some(&"acme-qwen3.8-27b-ft"));
        assert!(joined.ids().contains(&"qwen3.8-27b-16g"));
    }

    /// The class-aware step's answer, as the id that serves `id` on a card
    /// of class `card` or the refusal's words.
    fn resolved<'a>(catalog: &'a Catalog, id: &'a str, card: Option<CardClass>) -> Result<&'a str, String> {
        match catalog.answering(id, card) {
            Ok(Some(entry)) => Ok(entry.id.as_str()),
            Ok(None) => Ok(id),
            Err(refusal) => Err(refusal.to_string()),
        }
    }

    /// AC 2: a catalog must never write outside the destination directory —
    /// a separator of either kind, `..`, an absolute path or a drive.
    #[test]
    fn a_file_name_that_is_not_a_plain_name_is_refused() {
        for name in ["../escape.ninfer", "..", ".", "sub/acme_ft.ninfer", "sub\\acme_ft.ninfer", "/etc/acme.ninfer", "C:acme.ninfer", "C:\\models\\acme.ninfer"] {
            // Single-quoted: YAML reads no escape in it, so a `\` stays one.
            let text = OPERATOR.replacen("{ name: acme_ft.ninfer,", &format!("{{ name: '{name}',"), 1);
            assert_ne!(text, OPERATOR);
            let err = operator(&text).expect_err(name).0;
            assert!(err.contains("not a plain file name") && err.contains("acme-qwen3.8-27b-ft") && err.contains(name), "{name}: {err}");
        }
    }

    /// The allowlist behind AC 2: every built-in file name passes, the
    /// shapes that escape a directory or name a Windows device do not.
    #[test]
    fn plain_name_is_an_allowlist() {
        for entry in Catalog::built_in().entries() {
            for file in &entry.files {
                assert!(plain_name(&file.name), "{}", file.name);
            }
        }
        for name in ["qwen3_8_27b_nvfp4full-v2.ninfer", "a.b.c", "x-1_2", "console.ninfer", "COM10", "LPT0"] {
            assert!(plain_name(name), "{name}");
        }
        for name in [
            "", ".", "..", "NUL", "nul", "con.ninfer", "Com1.tar.gz", "aux", "lpt9.x", "a.ninfer.", ".hidden", "a b", "a\tb", "a\u{7f}b", "../x", "/x", "C:x", "a/b", r"a\b",
            "a*b", "a?b", "a<b", "a|b", "a\"b", "caf\u{e9}",
        ] {
            assert!(!plain_name(name), "{name:?}");
        }
    }

    #[test]
    fn a_catalog_that_is_not_a_list_of_entries_is_refused() {
        for (text, says) in [("[]", "`models` list"), ("{}", "`models`"), ("models: {}", "`models`"), ("models:\n  - just-a-name\n", "entry 1")] {
            let err = operator(text).expect_err(text).0;
            assert!(err.contains(says), "{text}: {err}");
        }
        assert!(operator("models: [").unwrap_err().0.contains("not valid"));
        assert_eq!(operator("models: []").unwrap(), *Catalog::built_in(), "an empty operator catalog adds nothing");
    }

    /// A `download.catalog` naming a file that is not there refuses, naming
    /// the path and the field.
    #[test]
    fn a_catalog_path_that_cannot_be_read_is_refused() {
        use crate::config::file::testing::MemFiles;
        let files = MemFiles::with(&[("acme.json", &serde_json::to_string(&Format::Yaml.read(OPERATOR).unwrap()).unwrap())]);
        assert_eq!(load(&files, None).unwrap(), *Catalog::built_in());
        assert!(load(&files, Some(Path::new("acme.json"))).unwrap().entry("acme-qwen3.8-27b-ft").is_some(), "read as JSON by extension");
        let err = load(&files, Some(Path::new("missing.yaml"))).unwrap_err().0;
        assert!(err.contains("missing.yaml") && err.contains("download.catalog"), "{err}");
    }
    #[test]
    fn the_single_baseline_artifact_and_family_alias_are_pinned() {
        let catalog = Catalog::built_in();
        assert_eq!(catalog.ids(), ["qwen3.8-27b-16g"]);
        let entry = catalog.entry("qwen3.8-27b").unwrap();
        assert_eq!(entry.family(), Some(ModelFamily::Qwen38_27b));
        assert_eq!(entry.revision, "f5bbd2cc14c9ca573ef5553c100ac69410f1fe58");
        assert_eq!(entry.artifact, "qwen3_8_27b_trellis_sc30-v2.ninfer");
        assert_eq!(entry.files.len(), 2);
        assert_eq!((entry.files[0].bytes, entry.files[0].sha256.as_str()), (309495, "210ab54bfe8b04e5fc4337ee3755c340ea3f21239aa625c505aa2406645a340d"));
        assert_eq!((entry.files[1].bytes, entry.files[1].sha256.as_str()), (13379877124, "2c4c9ee4d6b25c08da6c2d85840126e73d120564e230f926b6230b5b95ee8b3c"));
        for card in [Some(CardClass::Gb16), None] {
            assert_eq!(catalog.answering("qwen3.8-27b", card).unwrap(), Some(entry));
            assert!(catalog.alias_holds("qwen3.8-27b", entry, Some(ModelFamily::Qwen38_27b), card).is_ok());
            assert!(catalog.alias_holds("qwen3.8-27b", entry, Some(ModelFamily::FlashNext), card).is_err());
        }
        for id in ["qwen3.8-flash-next", "ignis-image-2.1-turbo", "qwen3-asr-1.7b", "nemotron-3-diarization"] {
            assert!(catalog.entry(id).is_none(), "{id}");
        }
    }

    #[test]
    fn speech_and_diarizer_catalogs_are_refused() {
        for kind in ["speech", "diarizer"] {
            let text = OPERATOR.replacen("    revision:", &format!("    kind: {kind}\n    revision:"), 1);
            assert!(operator(&text).unwrap_err().0.contains("unsupported in this fork"));
        }
    }

}
