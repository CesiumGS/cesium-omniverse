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

    uint64_t baseColorTextureCount = 0;

    if (!disableTextures) {
        if (materialInfo.baseColorTexture.has_value()) {
            baseColorTextureCount++;
        }

        baseColorTextureCount += imageryLayerCount;
    }

    _hasVertexColors = materialInfo.hasVertexColors;
    _baseColorTextureCount = baseColorTextureCount;
}

uint64_t FabricMaterialDefinition::getBaseColorTextureCount() const {
    return _baseColorTextureCount;
}

bool FabricMaterialDefinition::hasBaseColorTextures() const {
    return _baseColorTextureCount > 0;
}

bool FabricMaterialDefinition::hasVertexColors() const {
    return _hasVertexColors;
}

// In C++ 20 we can use the default equality comparison (= default)
bool FabricMaterialDefinition::operator==(const FabricMaterialDefinition& other) const {
    if (_hasVertexColors != other._hasVertexColors) {
        return false;
    }

    if (_baseColorTextureCount != other._baseColorTextureCount) {
        return false;
    }

    return true;
}

} // namespace cesium::omniverse
