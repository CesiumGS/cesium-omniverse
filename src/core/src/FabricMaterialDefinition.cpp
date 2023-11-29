#include "cesium/omniverse/FabricMaterialDefinition.h"

#include "cesium/omniverse/GltfUtil.h"
#include "cesium/omniverse/MetadataUtil.h"

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <CesiumGltf/Model.h>

namespace cesium::omniverse {

namespace {
std::vector<FeatureIdType> filterFeatureIdTypes(const FeaturesInfo& featuresInfo, bool disableTextures) {
    auto featureIdTypes = getFeatureIdTypes(featuresInfo);

    if (disableTextures) {
        featureIdTypes.erase(
            std::remove(featureIdTypes.begin(), featureIdTypes.end(), FeatureIdType::TEXTURE), featureIdTypes.end());
    }

    return featureIdTypes;
}
} // namespace

FabricMaterialDefinition::FabricMaterialDefinition(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const MaterialInfo& materialInfo,
    const FeaturesInfo& featuresInfo,
    uint64_t imageryLayerCount,
    bool disableTextures,
    const pxr::SdfPath& tilesetMaterialPath)
    : _hasVertexColors(materialInfo.hasVertexColors)
    , _hasBaseColorTexture(disableTextures ? false : materialInfo.baseColorTexture.has_value())
    , _featureIdTypes(filterFeatureIdTypes(featuresInfo, disableTextures))
    , _imageryLayerCount(disableTextures ? 0 : imageryLayerCount)
    , _tilesetMaterialPath(tilesetMaterialPath)
    , _mdlInternalPropertyAttributePropertyTypes(
          MetadataUtil::getMdlInternalPropertyAttributePropertyTypes(model, primitive))
    , _mdlInternalPropertyTexturePropertyTypes(
          MetadataUtil::getMdlInternalPropertyTexturePropertyTypes(model, primitive)) {}

bool FabricMaterialDefinition::hasVertexColors() const {
    return _hasVertexColors;
}

bool FabricMaterialDefinition::hasBaseColorTexture() const {
    return _hasBaseColorTexture;
}

const std::vector<FeatureIdType>& FabricMaterialDefinition::getFeatureIdTypes() const {
    return _featureIdTypes;
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

const std::vector<MdlInternalPropertyType>&
FabricMaterialDefinition::getMdlInternalPropertyAttributePropertyTypes() const {
    return _mdlInternalPropertyAttributePropertyTypes;
}

const std::vector<MdlInternalPropertyType>&
FabricMaterialDefinition::getMdlInternalPropertyTexturePropertyTypes() const {
    return _mdlInternalPropertyTexturePropertyTypes;
}

// In C++ 20 we can use the default equality comparison (= default)
bool FabricMaterialDefinition::operator==(const FabricMaterialDefinition& other) const {
    return _hasVertexColors == other._hasVertexColors && _hasBaseColorTexture == other._hasBaseColorTexture &&
           _featureIdTypes == other._featureIdTypes && _imageryLayerCount == other._imageryLayerCount &&
           _tilesetMaterialPath == other._tilesetMaterialPath &&
           _mdlInternalPropertyAttributePropertyTypes == other._mdlInternalPropertyAttributePropertyTypes &&
           _mdlInternalPropertyTexturePropertyTypes == other._mdlInternalPropertyTexturePropertyTypes;
}

} // namespace cesium::omniverse
