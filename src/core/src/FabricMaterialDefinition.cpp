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
    bool disableTextures,
    const pxr::SdfPath& tilesetMaterialPath)
    : _hasVertexColors(materialInfo.hasVertexColors)
    , _hasBaseColorTexture(disableTextures ? false : materialInfo.baseColorTexture.has_value())
    , _imageryLayerCount(disableTextures ? 0 : imageryLayerCount)
    , _tilesetMaterialPath(tilesetMaterialPath) {}

bool FabricMaterialDefinition::hasVertexColors() const {
    return _hasVertexColors;
}

bool FabricMaterialDefinition::hasBaseColorTexture() const {
    return _hasBaseColorTexture;
}

uint64_t FabricMaterialDefinition::getImageryLayerCount() const {
    return _imageryLayerCount;
}

bool FabricMaterialDefinition::hasTilesetMaterial() const {
    return !_tilesetMaterialPath.IsEmpty();
}

const pxr::SdfPath& FabricMaterialDefinition::getTilesetMaterialPath() const {
    return _tilesetMaterialPath;
}

// In C++ 20 we can use the default equality comparison (= default)
bool FabricMaterialDefinition::operator==(const FabricMaterialDefinition& other) const {
    return _hasVertexColors == other._hasVertexColors && _hasBaseColorTexture == other._hasBaseColorTexture &&
           _imageryLayerCount == other._imageryLayerCount && _tilesetMaterialPath == other._tilesetMaterialPath;
}

} // namespace cesium::omniverse
