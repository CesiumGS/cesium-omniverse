#pragma once

#include <omni/fabric/AttrNameAndType.h>

namespace omni::fabric {
class StageReaderWriter;
}

namespace cesium::omniverse {

class Context;

class FabricAttributesBuilder {
  public:
    FabricAttributesBuilder(Context* pContext);
    ~FabricAttributesBuilder() = default;
    FabricAttributesBuilder(const FabricAttributesBuilder&) = delete;
    FabricAttributesBuilder& operator=(const FabricAttributesBuilder&) = delete;
    FabricAttributesBuilder(FabricAttributesBuilder&&) noexcept = default;
    FabricAttributesBuilder& operator=(FabricAttributesBuilder&&) noexcept = default;

    void addAttribute(const omni::fabric::Type& type, const omni::fabric::Token& name);
    void createAttributes(const omni::fabric::Path& path) const;

  private:
    static const uint64_t MAX_ATTRIBUTES{30};
    uint64_t _size{0};
    std::array<omni::fabric::AttrNameAndType, MAX_ATTRIBUTES> _attributes;
    Context* _pContext;
};

} // namespace cesium::omniverse
