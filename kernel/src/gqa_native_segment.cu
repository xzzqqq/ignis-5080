#define IGNIS_GQA_TRANSIENT_WORKSPACE 1
#define IGNIS_GQA_SEGMENT_FN(name) ignis_gqa_native_segment_##name
#define IGNIS_GQA_ATTENTION_FN(name) ignis_gqa_native_##name
#include "gqa_token_mixer_impl.cuh"
#undef IGNIS_GQA_ATTENTION_FN
#undef IGNIS_GQA_SEGMENT_FN
#undef IGNIS_GQA_TRANSIENT_WORKSPACE
