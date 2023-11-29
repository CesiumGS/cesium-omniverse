#pragma once

#include "cesium/omniverse/DataType.h"
#include "cesium/omniverse/FabricTexture.h"
#include "cesium/omniverse/GltfUtil.h"
#include "cesium/omniverse/LoggerSink.h"

#include <CesiumGltf/ExtensionExtMeshFeatures.h>
#include <CesiumGltf/ExtensionMeshPrimitiveExtStructuralMetadata.h>
#include <CesiumGltf/ExtensionModelExtStructuralMetadata.h>
#include <CesiumGltf/PropertyAttribute.h>
#include <CesiumGltf/PropertyAttributeView.h>
#include <CesiumGltf/PropertyTableView.h>
#include <CesiumGltf/PropertyTexture.h>
#include <CesiumGltf/PropertyTextureView.h>

namespace cesium::omniverse::MetadataUtil {

template <DataType T> struct StyleablePropertyInfo {
    std::optional<GetNativeType<getTransformedType<T>()>> offset;
    std::optional<GetNativeType<getTransformedType<T>()>> scale;
    std::optional<GetNativeType<getTransformedType<T>()>> min;
    std::optional<GetNativeType<getTransformedType<T>()>> max;
    bool required;
    std::optional<GetNativeType<T>> noData;
    std::optional<GetNativeType<getTransformedType<T>()>> defaultValue;
};

template <DataType T> struct StyleablePropertyAttributePropertyInfo {
    static constexpr auto Type = T;
    std::string attribute;
    StyleablePropertyInfo<T> propertyInfo;
};

template <DataType T> struct StyleablePropertyTexturePropertyInfo {
    static constexpr auto Type = T;
    TextureInfo textureInfo;
    uint64_t textureIndex;
    StyleablePropertyInfo<T> propertyInfo;
};

template <DataType T> struct StyleablePropertyTablePropertyInfo {
    static constexpr auto Type = T;
    uint64_t featureIdSetIndex;
    StyleablePropertyInfo<T> propertyInfo;
};

template <typename Callback>
void forEachPropertyAttributeProperty(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    Callback&& callback) {

    const auto pStructuralMetadataModel = model.getExtension<CesiumGltf::ExtensionModelExtStructuralMetadata>();
    if (!pStructuralMetadataModel) {
        return;
    }

    const auto pStructuralMetadataPrimitive =
        primitive.getExtension<CesiumGltf::ExtensionMeshPrimitiveExtStructuralMetadata>();
    if (!pStructuralMetadataPrimitive) {
        return;
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
                "Property attribute is invalid and will be ignored. Status code: {}",
                static_cast<int>(propertyAttributeView.status()));
            continue;
        }

        propertyAttributeView.forEachProperty(
            primitive,
            [callback = std::forward<Callback>(callback),
             &propertyAttributeView,
             &pStructuralMetadataModel,
             &pPropertyAttribute](const std::string& propertyId, auto propertyAttributePropertyView) {
                if (propertyAttributePropertyView.status() != CesiumGltf::PropertyAttributePropertyViewStatus::Valid) {
                    CESIUM_LOG_WARN(
                        "Property \"{}\" is invalid and will be ignored. Status code: {}",
                        propertyId,
                        static_cast<int>(propertyAttributePropertyView.status()));
                    return;
                }

                const auto& schema = pStructuralMetadataModel->schema;
                if (!schema.has_value()) {
                    CESIUM_LOG_WARN("No schema found. Property \"{}\" will be ignored.", propertyId);
                    return;
                }

                const auto pClassDefinition = propertyAttributeView.getClass();
                if (!pClassDefinition) {
                    CESIUM_LOG_WARN("No class found. Property \"{}\" will be ignored.", propertyId);
                    return;
                }

                const auto pClassProperty = propertyAttributeView.getClassProperty(propertyId);
                if (!pClassProperty) {
                    CESIUM_LOG_WARN("No class property found. Property \"{}\" will be ignored.", propertyId);
                    return;
                }

                const auto& propertyAttributeProperty = pPropertyAttribute->properties.at(propertyId);

                callback(
                    propertyId,
                    schema.value(),
                    *pClassDefinition,
                    *pClassProperty,
                    *pPropertyAttribute,
                    propertyAttributeProperty,
                    propertyAttributeView,
                    propertyAttributePropertyView);
            });
    }
}

template <typename Callback>
void forEachPropertyTextureProperty(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    Callback&& callback) {

    const auto pStructuralMetadataModel = model.getExtension<CesiumGltf::ExtensionModelExtStructuralMetadata>();
    if (!pStructuralMetadataModel) {
        return;
    }

    const auto pStructuralMetadataPrimitive =
        primitive.getExtension<CesiumGltf::ExtensionMeshPrimitiveExtStructuralMetadata>();
    if (!pStructuralMetadataPrimitive) {
        return;
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
                "Property texture is invalid and will be ignored. Status code: {}",
                static_cast<int>(propertyTextureView.status()));

            continue;
        }

        propertyTextureView.forEachProperty(
            [callback = std::forward<Callback>(callback),
             &propertyTextureView,
             &pStructuralMetadataModel,
             &pPropertyTexture](const std::string& propertyId, auto propertyTexturePropertyView) {
                if (propertyTexturePropertyView.status() != CesiumGltf::PropertyTexturePropertyViewStatus::Valid) {
                    CESIUM_LOG_WARN(
                        "Property \"{}\" is invalid and will be ignored. Status code: {}",
                        propertyId,
                        static_cast<int>(propertyTexturePropertyView.status()));
                    return;
                }

                const auto& schema = pStructuralMetadataModel->schema;
                if (!schema.has_value()) {
                    CESIUM_LOG_WARN("No schema found. Property \"{}\" will be ignored.", propertyId);
                    return;
                }

                const auto pClassDefinition = propertyTextureView.getClass();
                if (!pClassDefinition) {
                    CESIUM_LOG_WARN("No class found. Property \"{}\" will be ignored.", propertyId);
                    return;
                }

                const auto pClassProperty = propertyTextureView.getClassProperty(propertyId);
                if (!pClassProperty) {
                    CESIUM_LOG_WARN("No class property found. Property \"{}\" will be ignored.", propertyId);
                    return;
                }

                if (!propertyTexturePropertyView.getImage()) {
                    CESIUM_LOG_WARN("No image found. Property \"{}\" will be ignored.", propertyId);
                    return;
                }

                const auto& propertyTextureProperty = pPropertyTexture->properties.at(propertyId);

                callback(
                    propertyId,
                    schema.value(),
                    *pClassDefinition,
                    *pClassProperty,
                    *pPropertyTexture,
                    propertyTextureProperty,
                    propertyTextureView,
                    propertyTexturePropertyView);
            });
    }
}

template <typename Callback>
void forEachPropertyTableProperty(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    Callback&& callback) {

    const auto pStructuralMetadataModel = model.getExtension<CesiumGltf::ExtensionModelExtStructuralMetadata>();
    if (!pStructuralMetadataModel) {
        return;
    }

    const auto pMeshFeatures = primitive.getExtension<CesiumGltf::ExtensionExtMeshFeatures>();
    if (!pMeshFeatures) {
        return;
    }

    for (uint64_t i = 0; i < pMeshFeatures->featureIds.size(); i++) {
        const auto featureIdSetIndex = i;
        const auto& featureId = pMeshFeatures->featureIds[i];
        if (featureId.propertyTable.has_value()) {
            const auto pPropertyTable = model.getSafe(
                &pStructuralMetadataModel->propertyTables, static_cast<int32_t>(featureId.propertyTable.value()));
            if (!pPropertyTable) {
                CESIUM_LOG_WARN("Property table index {} is out of range.", featureId.propertyTable.value());
                continue;
            }

            const auto propertyTableView = CesiumGltf::PropertyTableView(model, *pPropertyTable);
            if (propertyTableView.status() != CesiumGltf::PropertyTableViewStatus::Valid) {
                CESIUM_LOG_WARN(
                    "Property table is invalid and will be ignored. Status code: {}",
                    static_cast<int>(propertyTableView.status()));

                continue;
            }

            propertyTableView.forEachProperty(
                [callback = std::forward<Callback>(callback),
                 &propertyTableView,
                 &pStructuralMetadataModel,
                 &pPropertyTable,
                 featureIdSetIndex](const std::string& propertyId, auto propertyTablePropertyView) {
                    if (propertyTablePropertyView.status() != CesiumGltf::PropertyTablePropertyViewStatus::Valid) {
                        CESIUM_LOG_WARN(
                            "Property \"{}\" is invalid and will be ignored. Status code: {}",
                            propertyId,
                            static_cast<int>(propertyTablePropertyView.status()));
                        return;
                    }

                    const auto& schema = pStructuralMetadataModel->schema;
                    if (!schema.has_value()) {
                        CESIUM_LOG_WARN("No schema found. Property \"{}\" will be ignored.", propertyId);
                        return;
                    }

                    const auto pClassDefinition = propertyTableView.getClass();
                    if (!pClassDefinition) {
                        CESIUM_LOG_WARN("No class found. Property \"{}\" will be ignored.", propertyId);
                        return;
                    }

                    const auto pClassProperty = propertyTableView.getClassProperty(propertyId);
                    if (!pClassProperty) {
                        CESIUM_LOG_WARN("No class property found. Property \"{}\" will be ignored.", propertyId);
                        return;
                    }

                    const auto& propertyTableProperty = pPropertyTable->properties.at(propertyId);

                    callback(
                        propertyId,
                        schema.value(),
                        *pClassDefinition,
                        *pClassProperty,
                        *pPropertyTable,
                        propertyTableProperty,
                        propertyTableView,
                        propertyTablePropertyView,
                        featureIdSetIndex);
                });
        }
    }
}

template <typename Callback>
void forEachStyleablePropertyAttributeProperty(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    Callback&& callback) {

    forEachPropertyAttributeProperty(
        model,
        primitive,
        [callback = std::forward<Callback>(callback)](
            const std::string& propertyId,
            [[maybe_unused]] const CesiumGltf::Schema& schema,
            [[maybe_unused]] const CesiumGltf::Class& classDefinition,
            [[maybe_unused]] const CesiumGltf::ClassProperty& classProperty,
            [[maybe_unused]] const CesiumGltf::PropertyAttribute& propertyAttribute,
            const CesiumGltf::PropertyAttributeProperty& propertyAttributeProperty,
            [[maybe_unused]] const CesiumGltf::PropertyAttributeView& propertyAttributeView,
            auto propertyAttributePropertyView) {
            using RawType = decltype(propertyAttributePropertyView.getRaw(0));
            using TransformedType = typename std::decay_t<decltype(propertyAttributePropertyView.get(0))>::value_type;
            constexpr auto type = getTypeReverse<RawType, TransformedType>();

            if constexpr (isMatrix<type>()) {
                CESIUM_LOG_WARN(
                    "Matrix properties are not supported for styling. Property \"{}\" will be ignored.", propertyId);
                return;
            } else {
                const auto& attribute = propertyAttributeProperty.attribute;

                // For some reason the static cast is needed in MSVC
                const auto propertyInfo = StyleablePropertyInfo<static_cast<cesium::omniverse::DataType>(type)>{
                    propertyAttributePropertyView.offset(),
                    propertyAttributePropertyView.scale(),
                    propertyAttributePropertyView.min(),
                    propertyAttributePropertyView.max(),
                    propertyAttributePropertyView.required(),
                    propertyAttributePropertyView.noData(),
                    propertyAttributePropertyView.defaultValue(),
                };

                const auto styleableProperty =
                    StyleablePropertyAttributePropertyInfo<static_cast<cesium::omniverse::DataType>(type)>{
                        attribute,
                        propertyInfo,
                    };

                callback(propertyId, propertyAttributePropertyView, styleableProperty);
            }
        });
}

template <typename T, typename F> constexpr auto hasMemberImpl(F&& f) -> decltype(f(std::declval<T>()), true) {
    return true;
}

template <typename> constexpr bool hasMemberImpl(...) {
    return false;
}

#define HAS_MEMBER(T, EXPR) hasMemberImpl<T>([](auto&& obj) -> decltype(obj.EXPR) {})

template <typename Callback>
void forEachStyleablePropertyTextureProperty(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    Callback&& callback) {

    forEachPropertyTextureProperty(
        model,
        primitive,
        [callback = std::forward<Callback>(callback), &model](
            const std::string& propertyId,
            [[maybe_unused]] const CesiumGltf::Schema& schema,
            [[maybe_unused]] const CesiumGltf::Class& classDefinition,
            [[maybe_unused]] const CesiumGltf::ClassProperty& classProperty,
            [[maybe_unused]] const CesiumGltf::PropertyTexture& propertyTexture,
            const CesiumGltf::PropertyTextureProperty& propertyTextureProperty,
            [[maybe_unused]] const CesiumGltf::PropertyTextureView& propertyTextureView,
            auto propertyTexturePropertyView) {
            using RawType = decltype(propertyTexturePropertyView.getRaw(0.0, 0.0));
            using TransformedType =
                typename std::decay_t<decltype(propertyTexturePropertyView.get(0.0, 0.0))>::value_type;
            constexpr auto IsArray = HAS_MEMBER(RawType, size());

            if constexpr (IsArray) {
                CESIUM_LOG_WARN(
                    "Array properties are not supported for styling. Property \"{}\" will be ignored.", propertyId);
                return;
            } else {
                constexpr auto type = getTypeReverse<RawType, TransformedType>();

                if constexpr (getComponentByteLength<type>() > 1) {
                    CESIUM_LOG_WARN(
                        "Only 8-bit per-component property texture properties are supported for styling. Property "
                        "\"{}\" will be ignored.",
                        propertyId);
                    return;
                } else {
                    const auto textureInfo = GltfUtil::getPropertyTexturePropertyInfo(model, propertyTextureProperty);

                    if (textureInfo.channels.size() != getComponentCount<type>()) {
                        CESIUM_LOG_WARN(
                            "Properties with components that are packed across multiple texture channels are not "
                            "supported for styling. Property \"{}\" will be ignored.",
                            propertyId);
                        return;
                    }

                    if (textureInfo.channels.size() > 4) {
                        CESIUM_LOG_WARN(
                            "Properties with more than four channels are not supported for styling. Property \"{}\" "
                            "will be ignored.",
                            propertyId);
                        return;
                    }

                    const auto propertyInfo = StyleablePropertyInfo<static_cast<cesium::omniverse::DataType>(type)>{
                        propertyTexturePropertyView.offset(),
                        propertyTexturePropertyView.scale(),
                        propertyTexturePropertyView.min(),
                        propertyTexturePropertyView.max(),
                        propertyTexturePropertyView.required(),
                        propertyTexturePropertyView.noData(),
                        propertyTexturePropertyView.defaultValue(),
                    };

                    const auto styleableProperty =
                        StyleablePropertyTexturePropertyInfo<static_cast<cesium::omniverse::DataType>(type)>{
                            textureInfo,
                            static_cast<uint64_t>(propertyTextureProperty.index),
                            propertyInfo,
                        };

                    callback(propertyId, propertyTexturePropertyView, styleableProperty);
                }
            }
        });
}

template <typename Callback>
void forEachStyleablePropertyTableProperty(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    Callback&& callback) {

    forEachPropertyTableProperty(
        model,
        primitive,
        [callback = std::forward<Callback>(callback), &model](
            const std::string& propertyId,
            [[maybe_unused]] const CesiumGltf::Schema& schema,
            [[maybe_unused]] const CesiumGltf::Class& classDefinition,
            [[maybe_unused]] const CesiumGltf::ClassProperty& classProperty,
            [[maybe_unused]] const CesiumGltf::PropertyTable& propertyTable,
            [[maybe_unused]] const CesiumGltf::PropertyTableProperty& propertyTableProperty,
            [[maybe_unused]] const CesiumGltf::PropertyTableView& propertyTableView,
            auto propertyTablePropertyView,
            uint64_t featureIdSetIndex) {
            using RawType = decltype(propertyTablePropertyView.getRaw(0));
            using TransformedType = typename std::decay_t<decltype(propertyTablePropertyView.get(0))>::value_type;
            constexpr auto IsArray = HAS_MEMBER(RawType, size());
            constexpr auto IsBoolean = std::is_same_v<RawType, bool>;
            constexpr auto IsString = std::is_same_v<RawType, std::string_view>;

            if constexpr (IsArray) {
                CESIUM_LOG_WARN(
                    "Array properties are not supported for styling. Property \"{}\" will be ignored.", propertyId);
                return;
            } else if constexpr (IsBoolean) {
                CESIUM_LOG_WARN(
                    "Boolean properties are not supported for styling. Property \"{}\" will be ignored.", propertyId);
                return;
            } else if constexpr (IsString) {
                CESIUM_LOG_WARN(
                    "String properties are not supported for styling. Property \"{}\" will be ignored.", propertyId);
                return;
            } else {
                constexpr auto type = getTypeReverse<RawType, TransformedType>();
                constexpr auto unnormalizedComponentType = getUnnormalizedComponentType<type>();

                if constexpr (isMatrix<type>()) {
                    CESIUM_LOG_WARN(
                        "Matrix properties are not supported for styling. Property \"{}\" will be ignored.",
                        propertyId);
                    return;
                } else if constexpr (unnormalizedComponentType == DataType::UINT32) {
                    CESIUM_LOG_WARN(
                        "UINT32 properties are not supported for styling due to potential precision loss. Property "
                        "\"{}\" will be ignored.",
                        propertyId);
                    return;
                } else if constexpr (unnormalizedComponentType == DataType::UINT64) {
                    CESIUM_LOG_WARN(
                        "UINT64 properties are not supported for styling due to potential precision loss. Property "
                        "\"{}\" will be ignored.",
                        propertyId);
                    return;
                } else if constexpr (unnormalizedComponentType == DataType::INT64) {
                    CESIUM_LOG_WARN(
                        "INT64 properties are not supported for styling due to potential precision loss. Property "
                        "\"{}\" will be ignored.",
                        propertyId);
                    return;
                } else {
                    if constexpr (unnormalizedComponentType == DataType::FLOAT64) {
                        CESIUM_LOG_WARN(
                            "64-bit float properties are converted to 32-bit floats for styling. Some precision loss "
                            "may occur for property \"{}\".",
                            propertyId);
                    }

                    const auto propertyInfo = StyleablePropertyInfo<static_cast<cesium::omniverse::DataType>(type)>{
                        propertyTablePropertyView.offset(),
                        propertyTablePropertyView.scale(),
                        propertyTablePropertyView.min(),
                        propertyTablePropertyView.max(),
                        propertyTablePropertyView.required(),
                        propertyTablePropertyView.noData(),
                        propertyTablePropertyView.defaultValue(),
                    };

                    const auto styleableProperty =
                        StyleablePropertyTablePropertyInfo<static_cast<cesium::omniverse::DataType>(type)>{
                            featureIdSetIndex,
                            propertyInfo,
                        };

                    callback(propertyId, propertyTablePropertyView, styleableProperty);
                }
            }
        });
}

std::vector<MdlInternalPropertyType> getMdlInternalPropertyAttributePropertyTypes(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive);

std::vector<MdlInternalPropertyType>
getMdlInternalPropertyTexturePropertyTypes(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

std::vector<MdlInternalPropertyType>
getMdlInternalPropertyTexturePropertyTypes(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

std::vector<MdlInternalPropertyType>
getMdlInternalPropertyTablePropertyTypes(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

std::vector<const CesiumGltf::ImageCesium*>
getPropertyTextureImages(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

std::unordered_map<uint64_t, uint64_t>
getPropertyTextureIndexMapping(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

std::vector<TextureData>
encodePropertyTables(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

} // namespace cesium::omniverse::MetadataUtil
