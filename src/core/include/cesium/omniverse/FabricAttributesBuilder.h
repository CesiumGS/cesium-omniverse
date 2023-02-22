#pragma once

#include <carb/flatcache/StageWithHistory.h>

namespace cesium::omniverse {

// This is a helper class for creating Fabric attributes.

// Somewhat annoyingly, stageInProgress.createAttributes takes an std::array instead of a gsl::span. This is fine if you
// know exactly which set of attributes to create but we need greater flexibility than that. For example, not all prims
// will have UV coordinates or materials. This class allows attributes to be added dynamically up to a hardcoded
// maximum count (MAX_ATTRIBUTES) and avoids heap allocations. The downside is that the implementation is quite ugly.

class FabricAttributesBuilder {
  public:
    void addAttribute(const carb::flatcache::Type& type, const carb::flatcache::TokenC& name);
    void createAttributes(const carb::flatcache::Path& path);

  private:
    static const uint64_t MAX_ATTRIBUTES = 30;
    uint64_t _size = 0;
    std::array<carb::flatcache::AttrNameAndType, MAX_ATTRIBUTES> _attributes;
};
} // namespace cesium::omniverse
