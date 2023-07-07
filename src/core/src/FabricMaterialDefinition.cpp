#include "cesium/omniverse/FabricMaterialDefinition.h"

#include "cesium/omniverse/GltfUtil.h"

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <CesiumGltf/Model.h>

namespace cesium::omniverse {

FabricMaterialDefinition::FabricMaterialDefinition(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    bool hasImagery,
    bool disableTextures) {

    const auto hasGltfMaterial = GltfUtil::hasMaterial(primitive);

    if (hasGltfMaterial) {
        const auto materialInfo = GltfUtil::getMaterialInfo(model, primitive);
        _hasBaseColorTexture = materialInfo.baseColorTexture.has_value();
    } else {
        _hasBaseColorTexture = false;
    }

    if (hasImagery) {
        _hasBaseColorTexture = true;
    }

    if (disableTextures) {
        _hasBaseColorTexture = false;
    }

    _hasVertexColors = GltfUtil::hasVertexColors(model, primitive, 0);
}

bool FabricMaterialDefinition::hasBaseColorTexture() const {
    return _hasBaseColorTexture;
}

bool FabricMaterialDefinition::hasVertexColors() const {
    return _hasVertexColors;
}

// In C++ 20 we can use the default equality comparison (= default)
bool FabricMaterialDefinition::operator==(const FabricMaterialDefinition& other) const {
    if (_hasBaseColorTexture != other._hasBaseColorTexture) {
        return false;
    }

    if (_hasVertexColors != other._hasVertexColors) {
        return false;
    }

    return true;
}

} // namespace cesium::omniverse
