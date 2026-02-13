#include "cesium/omniverse/FabricAttributesBuilder.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/UsdUtil.h"

#include <omni/fabric/SimStageWithHistory.h>

namespace cesium::omniverse {

FabricAttributesBuilder::FabricAttributesBuilder(Context* pContext)
    : _pContext(pContext) {}

void FabricAttributesBuilder::addAttribute(const omni::fabric::Type& type, const omni::fabric::Token& name) {
    assert(_size < MAX_ATTRIBUTES);
    _attributes[_size++] = omni::fabric::AttrNameAndType(type, name);
}

void FabricAttributesBuilder::createAttributes(const omni::fabric::Path& path) const {
    auto& fabricStage = _pContext->getFabricStage();
    fabricStage.createAttributes(path, {_attributes.data(), _size});
}
} // namespace cesium::omniverse
