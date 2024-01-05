#include "cesium/omniverse/FabricMaterialDescriptor.h"

#include "cesium/omniverse/FabricFeaturesInfo.h"
#include "cesium/omniverse/FabricFeaturesUtil.h"
#include "cesium/omniverse/FabricImageryLayersInfo.h"
#include "cesium/omniverse/FabricMaterialInfo.h"
#include "cesium/omniverse/FabricPropertyDescriptor.h"
#include "cesium/omniverse/GltfUtil.h"
#include "cesium/omniverse/MetadataUtil.h"

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <CesiumGltf/Model.h>

namespace cesium::omniverse {

namespace {
std::vector<FabricPropertyDescriptor> getStyleableProperties(
    Context* pContext,
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const PXR_NS::SdfPath& tilesetMaterialPath) {

    if (tilesetMaterialPath.IsEmpty()) {
        // Return early, don't call getStyleableProperties because it logs
        // warnings for unsupported properties. Those warnings don't matter
        // if you're not using a tileset material.
        return {};
    }

    return MetadataUtil::getStyleableProperties(pContext, model, primitive);
}
} // namespace

FabricMaterialDescriptor::FabricMaterialDescriptor(
    Context* pContext,
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const FabricMaterialInfo& materialInfo,
    const FabricFeaturesInfo& featuresInfo,
    const FabricImageryLayersInfo& imageryLayersInfo,
    const PXR_NS::SdfPath& tilesetMaterialPath)
    : _hasVertexColors(materialInfo.hasVertexColors)
    , _hasBaseColorTexture(materialInfo.baseColorTexture.has_value())
    , _featureIdTypes(FabricFeaturesUtil::getFeatureIdTypes(featuresInfo))
    , _imageryOverlayRenderMethods(imageryLayersInfo.overlayRenderMethods)
    , _tilesetMaterialPath(tilesetMaterialPath)
    , _styleableProperties(
          ::cesium::omniverse::getStyleableProperties(pContext, model, primitive, tilesetMaterialPath)) {}

bool FabricMaterialDescriptor::hasVertexColors() const {
    return _hasVertexColors;
}

bool FabricMaterialDescriptor::hasBaseColorTexture() const {
    return _hasBaseColorTexture;
}

const std::vector<FabricFeatureIdType>& FabricMaterialDescriptor::getFeatureIdTypes() const {
    return _featureIdTypes;
}

const std::vector<FabricOverlayRenderMethod>& FabricMaterialDescriptor::getImageryOverlayRenderMethods() const {
    return _imageryOverlayRenderMethods;
}

bool FabricMaterialDescriptor::hasTilesetMaterial() const {
    return !_tilesetMaterialPath.IsEmpty();
}

const PXR_NS::SdfPath& FabricMaterialDescriptor::getTilesetMaterialPath() const {
    return _tilesetMaterialPath;
}

const std::vector<FabricPropertyDescriptor>& FabricMaterialDescriptor::getStyleableProperties() const {
    return _styleableProperties;
}

// Make sure to update this function when adding new fields to the class
// In C++ 20 we can use the default equality comparison (= default)
bool FabricMaterialDescriptor::operator==(const FabricMaterialDescriptor& other) const {
    // clang-format off
    return _hasVertexColors == other._hasVertexColors &&
           _hasBaseColorTexture == other._hasBaseColorTexture &&
           _featureIdTypes == other._featureIdTypes &&
           _imageryOverlayRenderMethods == other._imageryOverlayRenderMethods &&
           _tilesetMaterialPath == other._tilesetMaterialPath &&
           _styleableProperties == other._styleableProperties;
    // clang-format on
}

} // namespace cesium::omniverse
