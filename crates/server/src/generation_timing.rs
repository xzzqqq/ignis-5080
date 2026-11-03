//! Model-thread phase boundaries, independent of HTTP output buffering.
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use ignis_core::SchedEvent;
use serde::Serialize;
use utoipa::ToSchema;

pub(crate) type TimingResult = Arc<OnceLock<GenerationTiming>>;

#[derive(Debug, Clone, Serialize, ToSchema)]
pub(crate) struct GenerationTiming {
    /// Submission to the final prefill chunk, including admission wait.
    pub prefill_ms: f64,
    /// Final prefill chunk to scheduler completion, including the FIRST
    /// decode round, draft/verify, sampling, KV work and scheduling.
    pub decode_ms: f64,
    pub elapsed_ms: f64,
    /// Scheduler output tokens, including reasoning and tool syntax. Like
    /// usage.completion_tokens, excludes the terminal EOS stripped by runtime;
    /// the round that encounters EOS is still inside decode_ms.
    /// The prefill-sampled anchor is committed by the first timed round.
    pub output_tokens: u32,
    pub computed_prefill_tokens: u32,
    /// More than one means re-prefill occurred: not a simple two-phase run.
    pub prefill_passes: u32,
    pub speculative: Option<Speculative>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub(crate) struct Speculative {
    pub rounds: u32,
    pub drafted: u32,
    pub accepted: u32,
    pub drafted_at: Vec<u32>,
    pub accepted_at: Vec<u32>,
}

pub(crate) struct GenerationClock {
    submitted: Instant,
    prefilled: Option<Instant>,
    computed: u32,
    passes: u32,
    result: TimingResult,
}

impl GenerationClock {
    pub fn new(submitted: Instant, result: TimingResult) -> Self {
        Self { submitted, prefilled: None, computed: 0, passes: 0, result }
    }

    pub fn observe(&mut self, event: &SchedEvent, now: Instant) {
        match event {
            SchedEvent::PrefillChunk { chunk_tokens, .. } => self.computed += chunk_tokens,
            SchedEvent::PrefillComplete { at, .. } => {
                self.prefilled.get_or_insert(*at);
                self.passes += 1;
            }
            SchedEvent::Done { tokens, spec, .. } => {
                if let Some(prefilled) = self.prefilled {
                    let ms = |a: Instant, b: Instant| b.duration_since(a).as_secs_f64() * 1000.0;
                    let _ = self.result.set(GenerationTiming {
                        prefill_ms: ms(self.submitted, prefilled),
                        decode_ms: ms(prefilled, now),
                        elapsed_ms: ms(self.submitted, now),
                        output_tokens: *tokens,
                        computed_prefill_tokens: self.computed,
                        prefill_passes: self.passes,
                        speculative: spec.map(|s| Speculative {
                            rounds: s.rounds, drafted: s.drafted, accepted: s.accepted,
                            drafted_at: s.drafted_at.to_vec(), accepted_at: s.accepted_at.to_vec(),
                        }),
                    });
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use ignis_core::{FinishReason, SpecCounters};

    #[test]
    fn the_first_round_and_buffered_tokens_are_inside_the_window() {
        let start = Instant::now();
        let prefilled = start + Duration::from_millis(10);
        let done = start + Duration::from_millis(50);
        let result = TimingResult::default();
        let mut clock = GenerationClock::new(start, result.clone());
        clock.observe(&SchedEvent::PrefillChunk {
            request: 0, chunk_tokens: 128, prefilled_tokens: 128, encode_micros: 0,
        }, done);
        // All events can arrive together AFTER the first speculative round.
        clock.observe(&SchedEvent::PrefillComplete { request: 0, at: prefilled }, done);
        // The runtime stripped a terminal EOS from the eight-token round:
        // seven output events, but the WHOLE round remains timed.
        for token in 0..7 {
            clock.observe(&SchedEvent::Token { request: 0, token }, done);
        }
        clock.observe(&SchedEvent::Done {
            request: 0, tokens: 7, reason: FinishReason::Stop,
            spec: Some(SpecCounters { rounds: 1, drafted: 7, accepted: 7, ..Default::default() }),
            readout: None, attention: None, drawn: None, thinking: None, reasoning_redirected_at: None,
        }, done);
        let t = result.get().unwrap();
        assert_eq!((t.prefill_ms, t.decode_ms, t.elapsed_ms), (10.0, 40.0, 50.0));
        assert_eq!((t.output_tokens, t.computed_prefill_tokens, t.prefill_passes), (7, 128, 1));
        assert_eq!(t.speculative.as_ref().unwrap().accepted, 7);
    }
}
