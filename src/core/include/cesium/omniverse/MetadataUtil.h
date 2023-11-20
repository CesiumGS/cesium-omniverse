#pragma once

#include "cesium/omniverse/DataType.h"
#include "cesium/omniverse/GltfUtil.h"
#include "cesium/omniverse/LoggerSink.h"

#include <CesiumGltf/ExtensionMeshPrimitiveExtStructuralMetadata.h>
#include <CesiumGltf/ExtensionModelExtStructuralMetadata.h>
#include <CesiumGltf/PropertyAttribute.h>
#include <CesiumGltf/PropertyAttributeView.h>
#include <CesiumGltf/PropertyTexture.h>
#include <CesiumGltf/PropertyTextureView.h>

namespace cesium::omniverse::MetadataUtil {

template <DataType T> struct StyleablePropertyAttributePropertyInfo {
    static constexpr auto Type = T;
    std::optional<GetTransformedType<T>> offset;
    std::optional<GetTransformedType<T>> scale;
    std::optional<GetTransformedType<T>> min;
    std::optional<GetTransformedType<T>> max;
    bool required;
    std::optional<GetRawType<T>> noData;
    std::optional<GetTransformedType<T>> defaultValue;
    std::string attribute;
};
template <DataType T> struct StyleablePropertyTexturePropertyInfo {
    static constexpr auto Type = T;
    std::optional<GetTransformedType<T>> offset;
    std::optional<GetTransformedType<T>> scale;
    std::optional<GetTransformedType<T>> min;
    std::optional<GetTransformedType<T>> max;
    bool required;
    std::optional<GetRawType<T>> noData;
    std::optional<GetTransformedType<T>> defaultValue;
    TextureInfo textureInfo;
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
            constexpr auto Type = GetTypeReverse<RawType, TransformedType>::Type;

            if (IsMatrix<Type>::value) {
                CESIUM_LOG_WARN(
                    "Matrix properties are not supported for styling. Property \"{}\" will be ignored.", propertyId);
                return;
            }

            const auto& attribute = propertyAttributeProperty.attribute;

            const auto styleableProperty = StyleablePropertyAttributePropertyInfo<Type>{
                propertyAttributePropertyView.offset(),
                propertyAttributePropertyView.scale(),
                propertyAttributePropertyView.min(),
                propertyAttributePropertyView.max(),
                propertyAttributePropertyView.required(),
                propertyAttributePropertyView.noData(),
                propertyAttributePropertyView.defaultValue(),
                attribute,
            };

            callback(propertyAttributePropertyView, styleableProperty);
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
            [[maybe_unused]] const std::string& propertyId,
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
            constexpr bool IsArray = HAS_MEMBER(RawType, size());

            if constexpr (IsArray) {
                CESIUM_LOG_WARN(
                    "Array properties are not supported for styling. Property \"{}\" will be ignored.", propertyId);
                return;
            } else {
                constexpr auto Type = GetTypeReverse<RawType, TransformedType>::Type;

                const auto textureInfo = GltfUtil::getPropertyTexturePropertyInfo(model, propertyTextureProperty);

                if (textureInfo.channels.size() != GetComponentCount<Type>::ComponentCount) {
                    CESIUM_LOG_WARN(
                        "Properties with components that are packed across multiple texture channels are not supported "
                        "for styling. Property \"{}\" will be ignored.",
                        propertyId);
                    return;
                }

                if (IsFloatingPoint<Type>::value) {
                    CESIUM_LOG_WARN(
                        "Float property texture properties are not supported for styling. Property \"{}\" will be "
                        "ignored.",
                        propertyId);
                }

                const auto styleableProperty = StyleablePropertyTexturePropertyInfo<Type>{
                    propertyTexturePropertyView.offset(),
                    propertyTexturePropertyView.scale(),
                    propertyTexturePropertyView.min(),
                    propertyTexturePropertyView.max(),
                    propertyTexturePropertyView.required(),
                    propertyTexturePropertyView.noData(),
                    propertyTexturePropertyView.defaultValue(),
                    textureInfo,
                };

                callback(propertyTexturePropertyView, styleableProperty);
            }
        });
}

std::vector<DataType>
getMdlPropertyAttributePropertyTypes(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

std::vector<DataType>
getMdlPropertyTexturePropertyTypes(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

std::vector<const CesiumGltf::ImageCesium*>
getPropertyTextureImages(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

} // namespace cesium::omniverse::MetadataUtil
