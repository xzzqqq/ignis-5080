#pragma once
// Candidate-only exploration knob. 0 reproduces the resident split policy.
// 42 repartitions actual windows (20160,70000] into 42 declared splits.
// Both producer and reducer consume the same policy; numerical differences
// are expected and reported by exploratory replay rather than rejected.
#define IGNIS_LAB_GQA_FORCED_SPLITS 0
