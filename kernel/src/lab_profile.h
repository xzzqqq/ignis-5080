// Optional semantic ranges for Kernel Lab. NVTX records submission identity;
// only the profiler's correlated GPU activity supplies execution durations.
#pragma once
#include <cstdint>
namespace ignis {
bool lab_profile_enabled();
bool lab_profile_available();
bool lab_profile_push(const char *label);
bool lab_profile_field(const char *field, const char *value);
bool lab_profile_layer(uint32_t layer, const char *kind);
void lab_profile_pop();
class LabProfileRange {
  bool active_;
public:
  explicit LabProfileRange(const char *label) : active_(lab_profile_push(label)) {}
  LabProfileRange(const char *field, const char *value) : active_(lab_profile_field(field,value)) {}
  LabProfileRange(uint32_t layer, const char *kind) : active_(lab_profile_layer(layer,kind)) {}
  ~LabProfileRange() {if(active_)lab_profile_pop();}
  LabProfileRange(const LabProfileRange &) = delete;
  LabProfileRange &operator=(const LabProfileRange &) = delete;
};
}
// The immediately invoked lambda preserves void/return values and evaluates
// each argument once. The range also closes when an operator throws.
#define IGNIS_PROFILE_OP(name, ...) \
  ([&]() -> decltype(auto) { ignis::LabProfileRange profile_op_range("op",name); return __VA_ARGS__; }())
