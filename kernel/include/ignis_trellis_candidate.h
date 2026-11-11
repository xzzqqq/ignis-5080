/* Experimental, trusted-code ABI. No service endpoint exposes this API.
 * Candidates own code only; all weights, streams and workspaces belong to the host.
 * Version 1 preserves ignis_trellis_linear.h's layouts and numerical contract.
 * Calls allocate nothing, do not retain pointers, and are graph-capturable.
 * prepare/query run outside capture; the host clears the entire workspace
 * between candidates, so no candidate state persists in it.
 * A candidate must support every Trellis shape/width the fixed model uses.
 */
#ifndef IGNIS_TRELLIS_CANDIDATE_H
#define IGNIS_TRELLIS_CANDIDATE_H
#include "ignis_trellis_linear.h"
#ifdef __cplusplus
extern "C" {
#endif
#define IGNIS_TRELLIS_CANDIDATE_ABI 1
struct ignis_trellis_candidate_api {
  uint32_t version, size, weight_size, workspace_size;
  const char *identity;
  int32_t (*prepare)(void);
  uint64_t (*workspace_bytes)(uint32_t, uint32_t, uint32_t, uint32_t);
  int32_t (*linear)(const ignis_trellis_weight *, const void *, uint32_t, void *, uint32_t,
                    uint32_t, const ignis_trellis_workspace *, void *);
  int32_t (*swiglu)(const ignis_trellis_weight *, const ignis_trellis_weight *, const void *,
                    uint32_t, void *, uint32_t, const ignis_trellis_workspace *, void *);
  const char *(*last_error)(void);
};
/* The sole exported plugin symbol returns an immutable descriptor. */
const struct ignis_trellis_candidate_api *ignis_trellis_candidate_v1(void);
struct ignis_model;
struct ignis_seq_pool;
/* Single model thread, empty pool, no concurrent CUDA work. path NULL restores
 * built-in code. On admission/capture failure old code and graphs survive.
 * Fatal CUDA errors require process restart. Never use with untrusted code. */
int32_t ignis_model_reload_trellis(struct ignis_model *, struct ignis_seq_pool *, const char *path);
const char *ignis_trellis_reload_error(void);
#ifdef __cplusplus
}
#endif
#endif
