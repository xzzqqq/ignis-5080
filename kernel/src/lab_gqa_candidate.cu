#include "lab_gqa_candidate_policy.cuh"
#define IGNIS_GQA_FN(name) ignis_lab_gqa_##name
#include "gqa_hq_decode_impl.cuh"
#undef IGNIS_GQA_FN
