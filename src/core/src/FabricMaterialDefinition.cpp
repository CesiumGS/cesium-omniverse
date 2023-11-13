#include "cesium/omniverse/FabricMaterialDefinition.h"

#include "cesium/omniverse/DataType.h"
#include "cesium/omniverse/GltfUtil.h"
#include "cesium/omniverse/LoggerSink.h"

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
getMdlPropertyAttributeTypes(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
    std::vector<DataType> mdlPropertyTypes;

    const auto pStructuralMetadataPrimitive =
        primitive.getExtension<CesiumGltf::ExtensionMeshPrimitiveExtStructuralMetadata>();
    if (!pStructuralMetadataPrimitive) {
        return {};
    }

    const auto pStructuralMetadataModel = model.getExtension<CesiumGltf::ExtensionModelExtStructuralMetadata>();
    if (!pStructuralMetadataModel) {
        return {};
    }

    for (const auto& propertyAttributeIndex : pStructuralMetadataPrimitive->propertyAttributes) {
        const auto pPropertyAttribute =
            model.getSafe(&pStructuralMetadataModel->propertyAttributes, static_cast<int32_t>(propertyAttributeIndex));
        if (!pPropertyAttribute) {
            CESIUM_LOG_WARN("Property attribute index {} is out of range.", propertyAttributeIndex);
            continue;
        }

        const auto propertyAttributeView = CesiumGltf::PropertyAttributeView(model, *pPropertyAttribute);
        if (propertyAttributeView.status() != CesiumGltf::PropertyAttributeViewStatus::Valid) {
            CESIUM_LOG_WARN(
                "Property attribute is invalid and will be ignored. Status code: {}", propertyAttributeView.status());
            continue;
        }

        propertyAttributeView.forEachProperty(
            primitive, [&mdlPropertyTypes]([[maybe_unused]] const std::string& propertyName, auto view) {
                if (view.status() != CesiumGltf::PropertyAttributePropertyViewStatus::Valid) {
                    CESIUM_LOG_WARN(
                        "Property \"{}\" is invalid and will be ignored. Status code: {}", propertyName, view.status());
                    return;
                }

                constexpr auto propertyType =
                    GetNativeTypeReverse<typename std::decay_t<decltype(view.get(0))>::value_type>::Type;
                const auto mdlPropertyType = getMdlPropertyType(propertyType);

                if (mdlPropertyType == DataType::UNKOWN) {
                    CESIUM_LOG_WARN(
                        "Matrix properties are not supported for styling. Property \"{}\" will be ignored.",
                        propertyName);
                    return;
                }

                mdlPropertyTypes.push_back(mdlPropertyType);
            });
    }

    // Sorting ensures that FabricMaterialDefinition equality checking is consistent
    std::sort(mdlPropertyTypes.begin(), mdlPropertyTypes.end());

    return mdlPropertyTypes;
}

template <typename T, typename F> constexpr auto hasMemberImpl(F&& f) -> decltype(f(std::declval<T>()), true) {
    return true;
}

template <typename> constexpr bool hasMemberImpl(...) {
    return false;
}

#define HAS_MEMBER(T, EXPR) hasMemberImpl<T>([](auto&& obj) -> decltype(obj.EXPR) {})

std::vector<DataType>
getMdlPropertyTextureTypes(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
    std::vector<DataType> mdlPropertyTypes;

    const auto pStructuralMetadataPrimitive =
        primitive.getExtension<CesiumGltf::ExtensionMeshPrimitiveExtStructuralMetadata>();
    if (!pStructuralMetadataPrimitive) {
        return {};
    }

    const auto pStructuralMetadataModel = model.getExtension<CesiumGltf::ExtensionModelExtStructuralMetadata>();
    if (!pStructuralMetadataModel) {
        return {};
    }

    for (const auto& propertyTextureIndex : pStructuralMetadataPrimitive->propertyTextures) {
        const auto pPropertyTexture =
            model.getSafe(&pStructuralMetadataModel->propertyTextures, static_cast<int32_t>(propertyTextureIndex));
        if (!pPropertyTexture) {
            CESIUM_LOG_WARN("Property texture index {} is out of range.", propertyTextureIndex);
            continue;
        }

        const auto propertyTextureView = CesiumGltf::PropertyTextureView(model, *pPropertyTexture);
        if (propertyTextureView.status() != CesiumGltf::PropertyTextureViewStatus::Valid) {
            CESIUM_LOG_WARN(
                "Property texture is invalid and will be ignored. Status code: {}", propertyTextureView.status());

            continue;
        }

        propertyTextureView.forEachProperty(
            [&mdlPropertyTypes]([[maybe_unused]] const std::string& propertyName, auto view) {
                if (view.status() != CesiumGltf::PropertyTexturePropertyViewStatus::Valid) {
                    CESIUM_LOG_WARN(
                        "Property \"{}\" is invalid and will be ignored. Status code: {}", propertyName, view.status());
                    return;
                }

                using ElementType = typename std::decay_t<decltype(view.get(0.0, 0.0))>::value_type;

                // Check if it's an ArrayPropertyView based on presence of size() function
                constexpr auto isArray = HAS_MEMBER(ElementType, size());

                if constexpr (isArray) {
                    constexpr auto innerType =
                        GetNativeTypeReverse<typename std::decay_t<decltype(ElementType{}[0])>>::Type;
                    constexpr auto isScalar = getComponentCount(innerType) == 1;

                    if constexpr (!isScalar) {
                        CESIUM_LOG_WARN(
                            "Array properties of vectors or matrices are not supported for styling. Property \"{}\" "
                            "will be ignored.",
                            propertyName);
                        return;
                    } else {
                        const auto arrayCount = view.arrayCount();
                        assert(arrayCount > 0);

                        if (arrayCount > 4) {
                            CESIUM_LOG_WARN(
                                "Array properties with count greater than 4 are not supported for styling. Property "
                                "\"{}\" will be ignored.",
                                propertyName);
                            return;
                        }

                        const auto propertyType = fromComponentTypeAndCount(innerType, arrayCount);
                        const auto mdlPropertyType = getMdlPropertyType(propertyType);

                        if (mdlPropertyType == DataType::UNKOWN) {
                            CESIUM_LOG_WARN(
                                "Matrix properties are not supported for styling. Property \"{}\" will be ignored.",
                                propertyName);
                            return;
                        }

                        mdlPropertyTypes.push_back(mdlPropertyType);
                    }
                } else {
                    const auto propertyType = GetNativeTypeReverse<ElementType>::Type;
                    const auto mdlPropertyType = getMdlPropertyType(propertyType);

                    if (mdlPropertyType == DataType::UNKOWN) {
                        CESIUM_LOG_WARN(
                            "Matrix properties are not supported for styling. Property \"{}\" will be ignored.",
                            propertyName);
                        return;
                    }

                    mdlPropertyTypes.push_back(mdlPropertyType);
                }
            });
    }

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
    , _mdlPropertyAttributeTypes(getMdlPropertyAttributeTypes(model, primitive))
    , _mdlPropertyTextureTypes(getMdlPropertyTextureTypes(model, primitive)) {}

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

// In C++ 20 we can use the default equality comparison (= default)
bool FabricMaterialDefinition::operator==(const FabricMaterialDefinition& other) const {
    return _hasVertexColors == other._hasVertexColors && _hasBaseColorTexture == other._hasBaseColorTexture &&
           _featureIdTypes == other._featureIdTypes && _imageryLayerCount == other._imageryLayerCount &&
           _tilesetMaterialPath == other._tilesetMaterialPath &&
           _mdlPropertyAttributeTypes == other._mdlPropertyAttributeTypes &&
           _mdlPropertyTextureTypes == other._mdlPropertyTextureTypes;
}

} // namespace cesium::omniverse
