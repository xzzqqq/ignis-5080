#pragma once
#include "ignis_trellis_candidate.h"
#include "ignis_gqa_candidate.h"
#include "ignis_segment_candidate.h"
#include <array>
#include "core/arena.h"
#include <memory>

namespace ignis {
struct SegmentWorkspaceSet { std::array<std::unique_ptr<ninfer::DeviceBuffer>,3> buffers; };
struct TrellisCandidate {
  void *library = nullptr;
  const ignis_trellis_candidate_api *api = nullptr;
  const ignis_gqa_candidate_api *gqa = nullptr;
  std::unique_ptr<ninfer::DeviceBuffer> gqa_workspace;
  const ignis_segment_candidate_api *segments=nullptr;
  SegmentWorkspaceSet segment_workspaces;
  ~TrellisCandidate();
};
}
