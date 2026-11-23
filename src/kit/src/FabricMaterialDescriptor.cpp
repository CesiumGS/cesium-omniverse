#include "cesium/omniverse/FabricMaterialDescriptor.h"

#include "cesium/omniverse/FeaturesInfo.h"
#include "cesium/omniverse/FeaturesUtil.h"
#include "cesium/omniverse/GltfUtil.h"
#include "cesium/omniverse/MaterialInfo.h"
#include "cesium/omniverse/MetadataUtil.h"
#include "cesium/omniverse/PropertyDescriptor.h"
#include "cesium/omniverse/RasterOverlaysInfo.h"

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
    const MaterialInfo& materialInfo,
    const FeaturesInfo& featuresInfo,
    const RasterOverlaysInfo& rasterOverlaysInfo,
    const pxr::SdfPath& tilesetMaterialPath)
    : _hasVertexColors(materialInfo.hasVertexColors)
    , _hasBaseColorTexture(materialInfo.baseColorTexture.has_value())
    , _featureIdTypes(FeaturesUtil::getFeatureIdTypes(featuresInfo))
    , _overlayRenderMethods(rasterOverlaysInfo.overlayRenderMethods)
    , _tilesetMaterialPath(tilesetMaterialPath) {
    // Ignore styleable properties unless the tileset has a material
    if (!_tilesetMaterialPath.IsEmpty()) {
        std::tie(_styleableProperties, _unsupportedPropertyWarnings) =
            MetadataUtil::getStyleableProperties(context, model, primitive);
    }
}

bool FabricMaterialDescriptor::hasVertexColors() const {
    return _hasVertexColors;
}

bool FabricMaterialDescriptor::hasBaseColorTexture() const {
    return _hasBaseColorTexture;
}

const std::vector<FeatureIdType>& FabricMaterialDescriptor::getFeatureIdTypes() const {
    return _featureIdTypes;
}

const std::vector<OverlayRenderMethod>& FabricMaterialDescriptor::getOverlayRenderMethods() const {
    return _overlayRenderMethods;
}

bool FabricMaterialDescriptor::hasTilesetMaterial() const {
    return !_tilesetMaterialPath.IsEmpty();
}

const pxr::SdfPath& FabricMaterialDescriptor::getTilesetMaterialPath() const {
    return _tilesetMaterialPath;
}

const std::vector<PropertyDescriptor>& FabricMaterialDescriptor::getStyleableProperties() const {
    return _styleableProperties;
}

const std::map<std::string, std::string>& FabricMaterialDescriptor::getUnsupportedPropertyWarnings() const {
    return _unsupportedPropertyWarnings;
}

} // namespace cesium::omniverse
