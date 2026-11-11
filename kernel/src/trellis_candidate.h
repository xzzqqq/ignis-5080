#pragma once
#include "ignis_trellis_candidate.h"
#include <memory>

namespace ignis {
struct TrellisCandidate {
  void *library = nullptr;
  const ignis_trellis_candidate_api *api = nullptr;
  ~TrellisCandidate();
};
}
