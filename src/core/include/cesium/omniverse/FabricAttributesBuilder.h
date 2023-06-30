#pragma once

#include <omni/fabric/SimStageWithHistory.h>

namespace cesium::omniverse {

class FabricAttributesBuilder {
  public:
    void addAttribute(const omni::fabric::Type& type, const omni::fabric::TokenC& name);
    void createAttributes(const omni::fabric::Path& path) const;

  private:
    static const uint64_t MAX_ATTRIBUTES = 30;
    uint64_t _size = 0;
    std::array<omni::fabric::AttrNameAndType, MAX_ATTRIBUTES> _attributes;
};
} // namespace cesium::omniverse
