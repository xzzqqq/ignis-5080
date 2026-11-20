//! The 27B shape the DFlash2 replay examples load at (GitHub #340), and the
//! command-line flags that move it. Kernel-free, so its tests run on CPU
//! (`tests/replay_shape.rs` includes this file); `replay_load.rs` plans and
//! loads it.
//!
//! Defaults preserve the 2026-10-24 experiment: the trellis body uses its
//! historical 16 GiB budget, HQ KV and 262,144-token context. These are pinned
//! replay defaults, not aliases for the current server profile. Explicit KV
//! and context flags support paired KV experiments without changing serving.
//! The NVFP4 weight body runs the same historical engine shape --
//! context, chunk, rope, window, head -- on the 5090's derived budget at its
//! eight lanes, so a replay on both differs only in the body and the card.

use ignis_artifact::Body27b;
use ignis_core::{KvFormat, ProposalHead, RopeScaling, Speculation, SpeculativeBackend, VramMode};

/// The 16 GB class's VRAM budget (spec 27b-16gb/05), every byte the process
/// holds included.
pub const SIXTEEN_GIB: u64 = 17_179_869_184;
/// `vram.qwen38.headroom_bytes` of the built-in profiles, `1200M`.
pub const QWEN38_HEADROOM: u64 = 1200 << 20;

/// What a load is asked for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoadShape {
    pub vram: VramMode,
    pub max_context: u32,
    pub kv_format: KvFormat,
    /// The decode lanes: the pool's slots and the widest round.
    pub lanes: u32,
    pub prefill_chunk: u32,
    /// The DFlash2 window (full head); 0 for a plain load.
    pub window: u32,
    pub rope: RopeScaling,
}

/// The flags that move a replay's shape off its body's, taken out of the
/// command line before positional arguments: lane/chunk controls and explicit
/// `--kv-format` / `--max-context` overrides for paired KV experiments.
/// Flags rather than the server's environment
/// (`IGNIS_MODEL_PREFILL_CHUNK`): a variable the shell happens to carry must
/// not move a leg unseen.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ShapeFlags {
    pub lanes: Option<u32>,
    pub prefill_chunk: Option<u32>,
    pub max_context: Option<u32>,
    pub kv_format: Option<KvFormat>,
}

impl ShapeFlags {
    /// The flags in `args`, removed from it; every other argument stays, in
    /// order.
    pub fn take(args: &mut Vec<String>) -> Result<Self, String> {
        let mut flags = Self::default();
        let mut i = 0;
        while i < args.len() {
            if args[i] == "--kv-format" {
                let flag = args.remove(i);
                if i == args.len() {
                    return Err(format!("{flag} needs a value"));
                }
                flags.kv_format = Some(KvFormat::parse(&args.remove(i))?);
                continue;
            }
            let slot = match args[i].as_str() {
                "--lanes" => &mut flags.lanes,
                "--prefill-chunk" => &mut flags.prefill_chunk,
                "--max-context" => &mut flags.max_context,
                _ => {
                    i += 1;
                    continue;
                }
            };
            let flag = args.remove(i);
            if i == args.len() {
                return Err(format!("{flag} needs a value"));
            }
            let raw = args.remove(i);
            *slot = Some(raw.parse().map_err(|e| format!("{flag} `{raw}`: {e}"))?);
        }
        Ok(flags)
    }
}

impl LoadShape {
    /// The shape `body`'s replay runs at: `rtx50-16gb`'s plan in 16 GiB for
    /// the trellis body, and the same engine at eight lanes on the 5090's
    /// derived budget for the NVFP4 one.
    pub fn for_body(body: Body27b) -> Self {
        let sixteen_gb = Self {
            vram: VramMode::Explicit { budget_bytes: SIXTEEN_GIB, allow_oversubscription: false },
            max_context: 262_144,
            kv_format: KvFormat::HqE8_2b,
            lanes: 2,
            prefill_chunk: 1024,
            window: 7,
            rope: RopeScaling::NONE,
        };
        match body {
            Body27b::Trellis => sixteen_gb,
            Body27b::Nvfp4 => Self { vram: VramMode::Derived { headroom_bytes: QWEN38_HEADROOM }, lanes: 8, ..sixteen_gb },
        }
    }

    /// Historical lane/chunk controls remain limited to the
    /// NVFP4 weight body: its historical `rtx5090` profile served 2048-token chunks since
    /// GitHub #341, and a smoke at the trellis body's two lanes pairs the two
    /// on more than the body. The trellis body loads as a 16 GB card does, at
    /// two lanes and 1024-token chunks, and is refused any other.
    pub fn new(body: Body27b, flags: ShapeFlags) -> Result<Self, String> {
        let shape = Self::for_body(body);
        let moved = Self {
            lanes: flags.lanes.unwrap_or(shape.lanes),
            prefill_chunk: flags.prefill_chunk.unwrap_or(shape.prefill_chunk),
            max_context: flags.max_context.unwrap_or(shape.max_context),
            kv_format: flags.kv_format.unwrap_or(shape.kv_format),
            ..shape
        };
        if body == Body27b::Trellis
            && (moved.lanes != shape.lanes || moved.prefill_chunk != shape.prefill_chunk)
        {
            return Err(format!(
                "the trellis body loads as a 16 GB card does, at {} lanes and {}-token chunks, not {} and {}",
                shape.lanes, shape.prefill_chunk, moved.lanes, moved.prefill_chunk
            ));
        }
        if moved.max_context == 0 || moved.max_context > moved.kv_format.gqa_max_context() {
            return Err(format!("{} context must be in 1..={}, got {}",
                moved.kv_format.as_str(), moved.kv_format.gqa_max_context(), moved.max_context));
        }
        Ok(moved)
    }

    /// A final forced round still evaluates all columns; never truncate a job
    /// to make it fit a selected context or overflow this host-side check.
    pub fn validate_job_context(&self, tokens: u64, columns: u32) -> Result<(), String> {
        if tokens.checked_add(u64::from(columns)).is_none_or(|need| need > u64::from(self.max_context)) {
            return Err(format!("a job's {tokens} tokens and a round's {columns} columns outgrow the load's {}-token context", self.max_context));
        }
        Ok(())
    }

    /// The load's DFlash2 speculation, full head; none for a plain load.
    pub fn speculation(&self) -> Option<Speculation> {
        (self.window > 0).then(|| {
            Speculation::new(SpeculativeBackend::Dflash2, self.window)
                .unwrap_or_else(|e| panic!("window {}: {e}", self.window))
                .with_proposal_head(ProposalHead::Full)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn the_flags_come_out_of_the_command_line_wherever_they_stand() {
        let mut line = args(&["exe", "--lanes", "2", "ARTIFACT", "MANIFEST", "--prefill-chunk", "2048", "OUT"]);
        let flags = ShapeFlags::take(&mut line).unwrap();
        assert_eq!(flags, ShapeFlags { lanes: Some(2), prefill_chunk: Some(2048), ..ShapeFlags::default() });
        assert_eq!(line, args(&["exe", "ARTIFACT", "MANIFEST", "OUT"]));

        let mut line = args(&["exe", "ARTIFACT"]);
        assert_eq!(ShapeFlags::take(&mut line).unwrap(), ShapeFlags::default());
        assert_eq!(line, args(&["exe", "ARTIFACT"]));
    }

    #[test]
    fn a_flag_without_a_number_is_refused() {
        let err = ShapeFlags::take(&mut args(&["exe", "ARTIFACT", "--lanes"])).unwrap_err();
        assert!(err.contains("--lanes needs a value"), "{err}");
        let err = ShapeFlags::take(&mut args(&["exe", "--prefill-chunk", "big", "ARTIFACT"])).unwrap_err();
        assert!(err.contains("--prefill-chunk `big`"), "{err}");
    }

    /// Owner, 2026-10-24: every trellis run loads as a 16 GB card would.
    #[test]
    fn the_trellis_body_loads_as_a_sixteen_gb_card_and_takes_no_other_shape() {
        let shape = LoadShape::new(Body27b::Trellis, ShapeFlags::default()).unwrap();
        assert_eq!(
            (shape.vram, shape.max_context, shape.lanes, shape.prefill_chunk, shape.window, shape.rope),
            (
                VramMode::Explicit { budget_bytes: 17_179_869_184, allow_oversubscription: false },
                262_144,
                2,
                1024,
                7,
                RopeScaling::NONE
            )
        );
        // Its own values, named, are its shape.
        let named = ShapeFlags { lanes: Some(2), prefill_chunk: Some(1024), ..ShapeFlags::default() };
        assert_eq!(LoadShape::new(Body27b::Trellis, named).unwrap(), shape);
        for flags in [ShapeFlags { lanes: Some(8), ..ShapeFlags::default() }, ShapeFlags { prefill_chunk: Some(2048), ..ShapeFlags::default() }] {
            let err = LoadShape::new(Body27b::Trellis, flags).unwrap_err();
            assert!(err.contains("as a 16 GB card does"), "{err}");
        }
    }

    #[test]
    fn the_nvfp4_body_runs_the_trellis_engine_shape_and_moves_by_flag() {
        let trellis = LoadShape::for_body(Body27b::Trellis);
        let shape = LoadShape::new(Body27b::Nvfp4, ShapeFlags::default()).unwrap();
        assert_eq!(shape, LoadShape { vram: VramMode::Derived { headroom_bytes: 1200 << 20 }, lanes: 8, ..trellis });
        let moved = LoadShape::new(Body27b::Nvfp4, ShapeFlags { lanes: Some(2), prefill_chunk: Some(2048), ..ShapeFlags::default() }).unwrap();
        assert_eq!(moved, LoadShape { lanes: 2, prefill_chunk: 2048, ..shape });
    }

    #[test]
    fn explicit_kv_and_context_flags_preserve_the_historical_defaults() {
        let mut line = args(&["exe", "--kv-format", "nvfp4", "ARTIFACT", "--max-context", "139264"]);
        let flags = ShapeFlags::take(&mut line).unwrap();
        assert_eq!(line, args(&["exe", "ARTIFACT"]));
        let shape = LoadShape::new(Body27b::Trellis, flags).unwrap();
        assert_eq!((shape.kv_format, shape.max_context, shape.lanes, shape.prefill_chunk),
                   (KvFormat::Nvfp4, 139_264, 2, 1024));
        let old = LoadShape::new(Body27b::Trellis, ShapeFlags::default()).unwrap();
        assert_eq!((old.kv_format, old.max_context), (KvFormat::HqE8_2b, 262_144));
        for format in [KvFormat::Bf16, KvFormat::HqE8_2b, KvFormat::Nvfp4] {
            let shape = LoadShape::new(Body27b::Trellis, ShapeFlags {
                kv_format: Some(format), max_context: Some(131_072), ..ShapeFlags::default()
            }).unwrap();
            assert!(shape.validate_job_context(131_064, 8).is_ok());
            assert!(shape.validate_job_context(131_065, 8).is_err());
            assert!(shape.validate_job_context(u64::MAX, 8).is_err());
            assert!(LoadShape::new(Body27b::Trellis, ShapeFlags {
                kv_format: Some(format), max_context: Some(format.gqa_max_context() + 1), ..ShapeFlags::default()
            }).is_err());
        }
    }

    #[test]
    fn invalid_kv_context_or_missing_values_are_refused() {
        for flags in [["--kv-format"].as_slice(), ["--max-context"].as_slice(),
                      ["--kv-format", "int4"].as_slice(), ["--max-context", "bad"].as_slice()] {
            assert!(ShapeFlags::take(&mut args(flags)).is_err());
        }
        assert!(LoadShape::new(Body27b::Trellis, ShapeFlags {
            max_context: Some(0), ..ShapeFlags::default()
        }).is_err());
    }

    #[test]
    fn the_window_is_dflash2_with_the_full_head_and_none_without_one() {
        let shape = LoadShape::for_body(Body27b::Trellis);
        let speculation = shape.speculation().expect("a window of 7");
        assert_eq!((speculation.backend(), speculation.draft_tokens()), (SpeculativeBackend::Dflash2, 7));
        assert_eq!(speculation.proposal_head(), ProposalHead::Full);
        assert_eq!(LoadShape { window: 0, ..shape }.speculation(), None);
    }
}
