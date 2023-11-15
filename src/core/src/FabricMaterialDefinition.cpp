#include "cesium/omniverse/FabricMaterialDefinition.h"

#include "cesium/omniverse/DataType.h"
#include "cesium/omniverse/GltfUtil.h"
#include "cesium/omniverse/LoggerSink.h"

#include <CesiumGltf/PropertyType.h>
#include <CesiumGltf/Schema.h>

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <CesiumGltf/ExtensionMeshPrimitiveExtStructuralMetadata.h>
#include <CesiumGltf/ExtensionModelExtStructuralMetadata.h>
#include <CesiumGltf/Model.h>
#include <CesiumGltf/PropertyAttributePropertyView.h>
#include <CesiumGltf/PropertyAttributeView.h>
#include <CesiumGltf/PropertyTexturePropertyView.h>
#include <CesiumGltf/PropertyTextureView.h>

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

std::vector<DataType>
gatherMdlPropertyAttributeTypes(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
    std::vector<DataType> mdlPropertyTypes;

    GltfUtil::forEachPropertyAttributeProperty(
        model,
        primitive,
        [&mdlPropertyTypes](
            [[maybe_unused]] const std::string& propertyId,
            const CesiumGltf::Schema& schema,
            const CesiumGltf::PropertyAttributeView& propertyAttributeView,
            [[maybe_unused]] auto propertyAttributePropertyView) {
            const auto pClassProperty = propertyAttributeView.getClassProperty(propertyId);
            if (!pClassProperty) {
                return;
            }

            const auto propertyType = getClassPropertyType(schema, *pClassProperty);

            if (propertyType == DataType::UNKNOWN) {
                // Shouldn't ever reach here, but print a warning just in case
                CESIUM_LOG_WARN("Unsupported property type. Property \"{}\" will be ignored.", propertyId);
                return;
            }

            if (getGroup(propertyType) == TypeGroup::MATRIX) {
                // Matrix types aren't supported for styling
                // If we want to add support in the future we can pack each row in a separate primvar and reassemble in MDL
                CESIUM_LOG_WARN(
                    "Matrix property attributes are not supported for styling. Property \"{}\" will be ignored.",
                    propertyId);
                return;
            }

            const auto mdlPropertyType = getMdlPropertyType(propertyType);

            if (mdlPropertyType == DataType::UNKNOWN) {
                // Shouldn't ever reach here, but print a warning just in case
                CESIUM_LOG_WARN("Unsupported property type. Property \"{}\" will be ignored.", propertyId);
                return;
            }

            mdlPropertyTypes.push_back(mdlPropertyType);
        });

    // Sorting ensures that FabricMaterialDefinition equality checking is consistent
    std::sort(mdlPropertyTypes.begin(), mdlPropertyTypes.end());

    return mdlPropertyTypes;
}

std::vector<DataType>
gatherMdlPropertyTextureTypes(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
    std::vector<DataType> mdlPropertyTypes;

    GltfUtil::forEachPropertyAttributeProperty(
        model,
        primitive,
        [&mdlPropertyTypes](
            [[maybe_unused]] const std::string& propertyId,
            const CesiumGltf::Schema& schema,
            const CesiumGltf::PropertyAttributeView& propertyTextureView,
            [[maybe_unused]] auto propertyTexturePropertyView) {
            const auto pClassProperty = propertyTextureView.getClassProperty(propertyId);
            if (!pClassProperty) {
                return;
            }

            const auto propertyType = getClassPropertyType(schema, *pClassProperty);

            if (propertyType == DataType::UNKNOWN) {
                // Will reach here if it's an array of vectors or an array with count > 4
                CESIUM_LOG_WARN("Unsupported property type. Property \"{}\" will be ignored.", propertyId);
                return;
            }

            const auto mdlPropertyType = getMdlPropertyType(propertyType);

            if (mdlPropertyType == DataType::UNKNOWN) {
                // Shouldn't ever reach here, but print a warning just in case
                CESIUM_LOG_WARN("Unsupported property type. Property \"{}\" will be ignored.", propertyId);
                return;
            }

            mdlPropertyTypes.push_back(mdlPropertyType);
        });

    // Sorting ensures that FabricMaterialDefinition equality checking is consistent
    std::sort(mdlPropertyTypes.begin(), mdlPropertyTypes.end());

    return mdlPropertyTypes;
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
    , _mdlPropertyAttributeTypes(gatherMdlPropertyAttributeTypes(model, primitive))
    , _mdlPropertyTextureTypes(gatherMdlPropertyTextureTypes(model, primitive)) {}

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

const std::vector<DataType>& FabricMaterialDefinition::getMdlPropertyAttributeTypes() const {
    return _mdlPropertyAttributeTypes;
}

const std::vector<DataType>& FabricMaterialDefinition::getMdlPropertyTextureTypes() const {
    return _mdlPropertyTextureTypes;
}

// In C++ 20 we can use the default equality comparison (= default)
bool FabricMaterialDefinition::operator==(const FabricMaterialDefinition& other) const {
    return _hasVertexColors == other._hasVertexColors && _hasBaseColorTexture == other._hasBaseColorTexture &&
           _featureIdTypes == other._featureIdTypes && _imageryLayerCount == other._imageryLayerCount &&
           _tilesetMaterialPath == other._tilesetMaterialPath &&
           _mdlPropertyAttributeTypes == other._mdlPropertyAttributeTypes &&
           _mdlPropertyTextureTypes == other._mdlPropertyTextureTypes;
}

} // namespace cesium::omniverse
