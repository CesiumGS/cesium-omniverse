#include "cesium/omniverse/FabricMaterialDefinition.h"

#include "cesium/omniverse/GltfUtil.h"

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <CesiumGltf/Model.h>

namespace cesium::omniverse {

FabricMaterialDefinition::FabricMaterialDefinition(
    const MaterialInfo& materialInfo,
    uint64_t imageryLayerCount,
    bool disableTextures) {

    auto hasBaseColorTexture = materialInfo.baseColorTexture.has_value();

    if (disableTextures) {
        hasBaseColorTexture = false;
        imageryLayerCount = 0;
    }

    _hasVertexColors = materialInfo.hasVertexColors;
    _hasBaseColorTexture = hasBaseColorTexture;
    _imageryLayerCount = imageryLayerCount;
}

bool FabricMaterialDefinition::hasVertexColors() const {
    return _hasVertexColors;
}

bool FabricMaterialDefinition::hasBaseColorTexture() const {
    return _hasBaseColorTexture;
}

uint64_t FabricMaterialDefinition::getImageryLayerCount() const {
    return _imageryLayerCount;
}

// In C++ 20 we can use the default equality comparison (= default)
bool FabricMaterialDefinition::operator==(const FabricMaterialDefinition& other) const {
    if (_hasVertexColors != other._hasVertexColors) {
        return false;
    }

    if (_hasBaseColorTexture != other._hasBaseColorTexture) {
        return false;
    }

    if (_imageryLayerCount != other._imageryLayerCount) {
        return false;
    }

    return true;
}

} // namespace cesium::omniverse
