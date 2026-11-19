#include "gqa_hq_decode_policy.cuh"
#define IGNIS_GQA_FN(name) ignis_gqa_native_##name
#include "gqa_hq_decode_impl.cuh"
#undef IGNIS_GQA_FN
