#include "cesium/omniverse/FabricMaterialDescriptor.h"

#include "cesium/omniverse/FabricFeaturesInfo.h"
#include "cesium/omniverse/FabricFeaturesUtil.h"
#include "cesium/omniverse/FabricMaterialInfo.h"
#include "cesium/omniverse/FabricPropertyDescriptor.h"
#include "cesium/omniverse/FabricRasterOverlaysInfo.h"
#include "cesium/omniverse/GltfUtil.h"
#include "cesium/omniverse/MetadataUtil.h"

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <CesiumGltf/Model.h>

namespace cesium::omniverse {

FabricMaterialDescriptor::FabricMaterialDescriptor(
    const Context& context,
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const FabricMaterialInfo& materialInfo,
    const FabricFeaturesInfo& featuresInfo,
    const FabricRasterOverlaysInfo& rasterOverlaysInfo,
    const pxr::SdfPath& tilesetMaterialPath)
    : _hasVertexColors(materialInfo.hasVertexColors)
    , _hasBaseColorTexture(materialInfo.baseColorTexture.has_value())
    , _featureIdTypes(FabricFeaturesUtil::getFeatureIdTypes(featuresInfo))
    , _rasterOverlayRenderMethods(rasterOverlaysInfo.overlayRenderMethods)
    , _tilesetMaterialPath(tilesetMaterialPath) {
    std::tie(_styleableProperties, _unsupportedPropertyWarnings) =
        MetadataUtil::getStyleableProperties(context, model, primitive);
}

bool FabricMaterialDescriptor::hasVertexColors() const {
    return _hasVertexColors;
}

bool FabricMaterialDescriptor::hasBaseColorTexture() const {
    return _hasBaseColorTexture;
}

const std::vector<FabricFeatureIdType>& FabricMaterialDescriptor::getFeatureIdTypes() const {
    return _featureIdTypes;
}

const std::vector<FabricOverlayRenderMethod>& FabricMaterialDescriptor::getRasterOverlayRenderMethods() const {
    return _rasterOverlayRenderMethods;
}

bool FabricMaterialDescriptor::hasTilesetMaterial() const {
    return !_tilesetMaterialPath.IsEmpty();
}

const pxr::SdfPath& FabricMaterialDescriptor::getTilesetMaterialPath() const {
    return _tilesetMaterialPath;
}

const std::vector<FabricPropertyDescriptor>& FabricMaterialDescriptor::getStyleableProperties() const {
    return _styleableProperties;
}

const std::map<std::string, std::string>& FabricMaterialDescriptor::getUnsupportedPropertyWarnings() const {
    return _unsupportedPropertyWarnings;
}

// Make sure to update this function when adding new fields to the class
bool FabricMaterialDescriptor::operator==(const FabricMaterialDescriptor& other) const {
    // clang-format off
    return _hasVertexColors == other._hasVertexColors &&
           _hasBaseColorTexture == other._hasBaseColorTexture &&
           _featureIdTypes == other._featureIdTypes &&
           _rasterOverlayRenderMethods == other._rasterOverlayRenderMethods &&
           _tilesetMaterialPath == other._tilesetMaterialPath &&
           _styleableProperties == other._styleableProperties;
           // _unsupportedPropertyWarnings is intentionally not checked because it adds unecessary overhead

    // clang-format on
}

} // namespace cesium::omniverse
