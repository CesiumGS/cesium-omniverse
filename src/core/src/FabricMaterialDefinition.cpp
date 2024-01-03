#include "cesium/omniverse/FabricMaterialDefinition.h"

#include "cesium/omniverse/GltfUtil.h"
#include "cesium/omniverse/MetadataUtil.h"

#include <vector>

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

std::vector<MetadataUtil::PropertyDefinition> getStyleableProperties(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const pxr::SdfPath& tilesetMaterialPath) {

    if (tilesetMaterialPath.IsEmpty()) {
        // Ignore properties if there's no tileset material
        return {};
    }

    return MetadataUtil::getStyleableProperties(model, primitive);
}
} // namespace

FabricMaterialDefinition::FabricMaterialDefinition(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const MaterialInfo& materialInfo,
    const FeaturesInfo& featuresInfo,
    const ImageryLayersInfo& imageryLayersInfo,
    bool disableTextures,
    const pxr::SdfPath& tilesetMaterialPath)
    : _hasVertexColors(materialInfo.hasVertexColors)
    , _hasBaseColorTexture(disableTextures ? false : materialInfo.baseColorTexture.has_value())
    , _featureIdTypes(filterFeatureIdTypes(featuresInfo, disableTextures))
    , _tilesetMaterialPath(tilesetMaterialPath)
    , _properties(getStyleableProperties(model, primitive, tilesetMaterialPath)) {
    _imageryLayerCount = imageryLayersInfo.imageryLayerCount;

    int layerNum = 0;
    for (auto layerType : imageryLayersInfo.overlayTypes) {
        switch (layerType) {
            case OverlayType::IMAGERY:
                _ionImageryLayerCount++;
                _ionImageryLayerIndices.push_back(layerNum);
                break;
            case OverlayType::POLYGON:
                _polygonImageryLayerCount++;
                _polygonImageryLayerIndices.push_back(layerNum);
                break;
        }
        layerNum++;
    }

    if (disableTextures) {
        _imageryLayerCount = 0;
        _ionImageryLayerCount = 0;
    }
}

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

uint64_t FabricMaterialDefinition::getPolygonImageryCount() const {
    return _polygonImageryLayerCount;
}

uint64_t FabricMaterialDefinition::getIonImageryCount() const {
    return _ionImageryLayerCount;
}

std::vector<int> FabricMaterialDefinition::getIonImageryLayerIndices() const {
    return _ionImageryLayerIndices;
}
std::vector<int> FabricMaterialDefinition::getPolygonImageryLayerIndices() const {
    return _polygonImageryLayerIndices;
}

bool FabricMaterialDefinition::hasTilesetMaterial() const {
    return !_tilesetMaterialPath.IsEmpty();
}

const pxr::SdfPath& FabricMaterialDefinition::getTilesetMaterialPath() const {
    return _tilesetMaterialPath;
}

const std::vector<MetadataUtil::PropertyDefinition>& FabricMaterialDefinition::getProperties() const {
    return _properties;
}

// In C++ 20 we can use the default equality comparison (= default)
bool FabricMaterialDefinition::operator==(const FabricMaterialDefinition& other) const {
    return _hasVertexColors == other._hasVertexColors && _hasBaseColorTexture == other._hasBaseColorTexture &&
           _featureIdTypes == other._featureIdTypes && _imageryLayerCount == other._imageryLayerCount &&
           _tilesetMaterialPath == other._tilesetMaterialPath && _properties == other._properties;
}

} // namespace cesium::omniverse
