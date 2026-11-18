#pragma once
// Structural experiment: 1 uses separate gate/up projections, stored BF16
// intermediates and a standalone SiLU-multiply kernel. This changes workspace
// layout/launch sequence and introduces extra rounding; 0 is the fused default.
#define IGNIS_LAB_MLP_SPLIT_SWIGLU 0
