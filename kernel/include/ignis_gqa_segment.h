/* C1/T8 Trellis + HQ attention sublayer. Only boundary tensors are exposed:
 * input residual -> input norm/projections/QK norm/RoPE/HQ attention/output
 * projection + residual -> output residual. The MLP is a separate segment.
 * Positions address KV; rope_positions independently address the RoPE table.
 * State uses the physical layout documented by ignis_gqa_candidate.h, with
 * complete planes so table row selection and untouched slots remain visible.
 */
#ifndef IGNIS_GQA_SEGMENT_H
#define IGNIS_GQA_SEGMENT_H
#include "ignis_segment_candidate.h"
#include "ignis_trellis_linear.h"
#ifdef __cplusplus
extern "C" {
#endif
struct ignis_gqa_segment_call {
  struct ignis_segment_profile profile;
  const void *input, *input_norm, *query_norm, *key_norm;
  ignis_trellis_weight q_proj, k_proj, v_proj, output_proj;
  double rope_inv_frequency[128];
  float rope_attention_factor;
  const int32_t *positions, *rope_positions, *valid_columns, *table_rows, *block_tables;
  uint32_t logical_pages, physical_pages;
  void *k_pages, *v_pages, *k_meta, *v_meta, *residual_k, *residual_v;
  uint32_t *ring;
  void *output;
};
int32_t ignis_lab_gqa_segment_prepare(void);
uint64_t ignis_lab_gqa_segment_workspace(const struct ignis_segment_profile *);
int32_t ignis_lab_gqa_segment_run(const struct ignis_gqa_segment_call *,void *,uint64_t,void *);
const char *ignis_lab_gqa_segment_error(void);
#ifdef __cplusplus
}
#endif
#endif
