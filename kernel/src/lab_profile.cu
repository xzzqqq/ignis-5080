#include "lab_profile.h"
#include <cstdlib>
#include <cstring>
#include <string>
#ifdef IGNIS_NVTX3_AVAILABLE
#include <nvtx3/nvToolsExt.h>
#endif
namespace ignis {
bool lab_profile_enabled() {
  static const bool enabled=[] {const char *v=std::getenv("IGNIS_LAB_PROFILE");return v && std::strcmp(v,"1")==0;}();
  return enabled;
}
bool lab_profile_available() {
#ifdef IGNIS_NVTX3_AVAILABLE
  return true;
#else
  return false;
#endif
}
bool lab_profile_push(const char *label) {
  if(!lab_profile_enabled() || !lab_profile_available())return false;
#ifdef IGNIS_NVTX3_AVAILABLE
  nvtxRangePushA(label);
#endif
  return true;
}
bool lab_profile_field(const char *field,const char *value) {
  if(!lab_profile_enabled() || !lab_profile_available())return false;
  const std::string label=std::string("ignis.profile.v1/")+field+"="+value;
  return lab_profile_push(label.c_str());
}
bool lab_profile_layer(uint32_t layer,const char *kind) {
  if(!lab_profile_enabled() || !lab_profile_available())return false;
  const std::string value=std::to_string(layer)+":"+kind;
  return lab_profile_field("layer",value.c_str());
}
void lab_profile_pop() {
#ifdef IGNIS_NVTX3_AVAILABLE
  nvtxRangePop();
#endif
}
}
extern "C" int32_t ignis_lab_profile_status() {
  return !ignis::lab_profile_enabled()?0:ignis::lab_profile_available()?1:-1;
}
extern "C" int32_t ignis_lab_profile_push(const char *label) {
  return label && ignis::lab_profile_push(label)?1:0;
}
extern "C" void ignis_lab_profile_pop() {ignis::lab_profile_pop();}
