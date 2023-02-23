#pragma once

#include <carb/flatcache/StageWithHistory.h>

namespace cesium::omniverse {

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
