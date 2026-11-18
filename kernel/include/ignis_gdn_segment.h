/* Verify GDN sublayer: input residual through output projection/residual.
 * The following MLP is outside this boundary. Persistent state is read-only;
 * all four ReplaySSM record planes are externally consumed liveouts. Invalid
 * key/value/gate record suffixes retain their incoming bytes.
 */
#ifndef IGNIS_GDN_SEGMENT_H
#define IGNIS_GDN_SEGMENT_H
#include "ignis_segment_candidate.h"
#include "ignis_trellis_linear.h"
#ifdef __cplusplus
extern "C" {
#endif
struct ignis_gdn_segment_call {
  struct ignis_segment_profile profile;
  struct ignis_trellis_weight qkv_weight,z_weight,output_weight;
  const void *input_norm,*convolution,*ab_projection,*a_log,*dt_bias,*gdn_norm;
  const void *residual_in,*conv_states,*recurrent_states;
  const int32_t *lane_slots,*valid_columns;
  void *residual_out,*conv_record,*key_record,*value_record,*gate_record;
};
int32_t ignis_lab_gdn_segment_prepare(void);
uint64_t ignis_lab_gdn_segment_workspace(const struct ignis_segment_profile *);
int32_t ignis_lab_gdn_segment_run(const struct ignis_gdn_segment_call *,void *,uint64_t,void *);
const char *ignis_lab_gdn_segment_error(void);
#ifdef __cplusplus
}
#endif
#endif
