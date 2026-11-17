// CPU-only contract for the optional range guard: instrumentation cannot
// change operator values, evaluations or exception unwinding.
#include "lab_profile.h"
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
void require(bool condition) {
  if(!condition) {std::cerr<<"profile range contract failed\n";std::abort();}
}
bool enabled=false;
std::vector<std::string> events;
}
namespace ignis {
bool lab_profile_enabled() {return enabled;}
bool lab_profile_available() {return true;}
bool lab_profile_push(const char *label) {
  if(!enabled)return false;
  events.emplace_back(label);return true;
}
bool lab_profile_field(const char *field,const char *value) {
  if(!enabled)return false;
  return lab_profile_push((std::string("ignis.profile.v1/")+field+"="+value).c_str());
}
bool lab_profile_layer(uint32_t layer,const char *kind) {
  return lab_profile_field("layer",(std::to_string(layer)+":"+kind).c_str());
}
void lab_profile_pop() {events.emplace_back("pop");}
}
int main() {
  int evaluations=0;
  require(IGNIS_PROFILE_OP("disabled",++evaluations)==1);
  require(events.empty());
  enabled=true;
  {
    ignis::LabProfileRange layer(63,"gqa");
    require(IGNIS_PROFILE_OP("output_head",++evaluations)==2);
    // Returning a reference and void are both legal operator signatures.
    int &alias=IGNIS_PROFILE_OP("reference",(evaluations));
    alias=7;
    IGNIS_PROFILE_OP("void",static_cast<void>(++evaluations));
    try {
      IGNIS_PROFILE_OP("failure",throw std::runtime_error("expected"));
      require(false);
    } catch(const std::runtime_error &) {}
  }
  require(evaluations==8);
  const std::vector<std::string> expected={
    "ignis.profile.v1/layer=63:gqa","ignis.profile.v1/op=output_head","pop",
    "ignis.profile.v1/op=reference","pop","ignis.profile.v1/op=void","pop",
    "ignis.profile.v1/op=failure","pop","pop"};
  require(events==expected);
}
