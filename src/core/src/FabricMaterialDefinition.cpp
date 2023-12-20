#include "cesium/omniverse/FabricMaterialDefinition.h"

#include "cesium/omniverse/GltfUtil.h"

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
    const MaterialInfo& materialInfo,
    const FeaturesInfo& featuresInfo,
    const ImageryLayersInfo& imageryLayersInfo,
    bool disableTextures,
    const pxr::SdfPath& tilesetMaterialPath)
    : _hasVertexColors(materialInfo.hasVertexColors)
    , _hasBaseColorTexture(disableTextures ? false : materialInfo.baseColorTexture.has_value())
    , _featureIdTypes(filterFeatureIdTypes(featuresInfo, disableTextures))
    , _tilesetMaterialPath(tilesetMaterialPath) {
        _imageryLayerCount = imageryLayersInfo.imageryLayerCount;
        for (auto layerType : imageryLayersInfo.overlayTypes) {
            switch (layerType) {
                case OverlayType::IMAGERY:
                    _ionImageryLayerCount++;
                break;
                case OverlayType::POLYGON:
                    _polygonImageryLayerCount++;
                break;
            }
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

bool FabricMaterialDefinition::hasTilesetMaterial() const {
    return !_tilesetMaterialPath.IsEmpty();
}

const pxr::SdfPath& FabricMaterialDefinition::getTilesetMaterialPath() const {
    return _tilesetMaterialPath;
}

// In C++ 20 we can use the default equality comparison (= default)
bool FabricMaterialDefinition::operator==(const FabricMaterialDefinition& other) const {
    return _hasVertexColors == other._hasVertexColors && _hasBaseColorTexture == other._hasBaseColorTexture &&
           _featureIdTypes == other._featureIdTypes && _imageryLayerCount == other._imageryLayerCount &&
           _tilesetMaterialPath == other._tilesetMaterialPath;
}

} // namespace cesium::omniverse
