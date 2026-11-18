/* Trusted offline Kernel Lab extension. Optional beside Trellis ABI v1.
 * Fixed profile v1: contiguous BF16 [256,24,8,1] Q/gate/output and
 * [256,4,8,1] K/V; I32 positions[8], optional valid_columns[1], table_rows[1].
 * HQ code/meta page-major planes [64|8,64,4,physical_pages], BF16 residual
 * [256,4,544,slots] and U32 ring[16,slots]. Tables I32[logical_pages,slots].
 * The host owns every allocation and stream. run allocates nothing, retains
 * no pointers and is graph-capturable. Candidate owns ALL attention launches,
 * partition/layout policy and reduction; workspace contents are unspecified.
 */
#ifndef IGNIS_GQA_CANDIDATE_H
#define IGNIS_GQA_CANDIDATE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define IGNIS_GQA_CANDIDATE_ABI 1
struct ignis_gqa_candidate_call {
  uint32_t version, size, max_visible_keys, logical_pages, physical_pages, slots;
  float scale;
  const void *q, *k, *v, *gate;
  const int32_t *positions, *valid_columns, *table_rows, *block_tables;
  void *k_pages, *v_pages, *k_meta, *v_meta, *residual_k, *residual_v;
  uint32_t *ring;
  void *output;
};
struct ignis_gqa_candidate_api {
  uint32_t version, size, call_size, reserved;
  const char *identity;
  int32_t (*prepare)(void);
  uint64_t (*workspace_bytes)(uint32_t max_visible_keys);
  int32_t (*run)(const struct ignis_gqa_candidate_call *, void *workspace, uint64_t bytes, void *stream);
  const char *(*last_error)(void);
};
const struct ignis_gqa_candidate_api *ignis_gqa_candidate_v1(void);
#ifdef __cplusplus
}
#endif
#endif
