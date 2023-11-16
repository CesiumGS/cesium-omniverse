#pragma once

#include "cesium/omniverse/DataType.h"
#include "cesium/omniverse/GltfUtil.h"

namespace cesium::omniverse::MetadataUtil {

template <DataType T> struct StyleablePropertyAttributePropertyInfo {
    static constexpr auto Type = T;
    std::string attribute;
    std::optional<GetTransformedType<T>> offset;
    std::optional<GetTransformedType<T>> scale;
    std::optional<GetTransformedType<T>> min;
    std::optional<GetTransformedType<T>> max;
    bool required;
    std::optional<GetRawType<T>> noData;
    std::optional<GetTransformedType<T>> defaultValue;
};

template <DataType T> struct StyleablePropertyTexturePropertyInfo {
    static constexpr auto Type = T;
    TextureInfo textureInfo;
    std::optional<GetTransformedType<Type>> offset;
    std::optional<GetTransformedType<Type>> scale;
    std::optional<GetTransformedType<Type>> min;
    std::optional<GetTransformedType<Type>> max;
    bool required;
    std::optional<GetRawType<Type>> noData;
    std::optional<GetTransformedType<Type>> defaultValue;
};

template <typename Callback>
void forEachPropertyAttributeProperty(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    Callback&& callback) {
    const auto pStructuralMetadataPrimitive =
        primitive.getExtension<CesiumGltf::ExtensionMeshPrimitiveExtStructuralMetadata>();
    if (!pStructuralMetadataPrimitive) {
        return;
    }

    const auto pStructuralMetadataModel = model.getExtension<CesiumGltf::ExtensionModelExtStructuralMetadata>();
    if (!pStructuralMetadataModel) {
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
             &pPropertyAttribute]([[maybe_unused]] const std::string& propertyId, auto propertyAttributePropertyView) {
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
    const auto pStructuralMetadataPrimitive =
        primitive.getExtension<CesiumGltf::ExtensionMeshPrimitiveExtStructuralMetadata>();
    if (!pStructuralMetadataPrimitive) {
        return;
    }

    const auto pStructuralMetadataModel = model.getExtension<CesiumGltf::ExtensionModelExtStructuralMetadata>();
    if (!pStructuralMetadataModel) {
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
             &pPropertyTexture]([[maybe_unused]] const std::string& propertyId, auto propertyTexturePropertyView) {
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
            const CesiumGltf::Schema& schema,
            const CesiumGltf::Class& classDefinition,
            const CesiumGltf::ClassProperty& classProperty,
            const CesiumGltf::PropertyAttribute& propertyAttribute,
            const CesiumGltf::PropertyAttributeProperty& propertyAttributeProperty,
            const CesiumGltf::PropertyAttributeView& propertyAttributeView,
            auto propertyAttributePropertyView) {
            using RawType = decltype(propertyAttributePropertyView.getRaw(0));
            using TransformedType = typename std::decay_t<decltype(propertyAttributePropertyView.get(0))>::value_type;
            constexpr auto type = GetType<RawType, TransformedType>::Type;

            if (IsMatrix<type>::value) {
                // Matrices are not supported
                CESIUM_LOG_WARN("Unsupported property type. Property \"{}\" will be ignored.", propertyId);
                return;
            }

            const auto& attribute = propertyAttributeProperty.attribute;

            const auto styleableProperty = StyleablePropertyAttributePropertyInfo<type>{
                attribute,
                propertyAttributePropertyView.offset(),
                propertyAttributePropertyView.scale(),
                propertyAttributePropertyView.min(),
                propertyAttributePropertyView.max(),
                propertyAttributePropertyView.required(),
                propertyAttributePropertyView.noData(),
                propertyAttributePropertyView.defaultValue(),
            };

            callback(
                propertyId,
                schema,
                classDefinition,
                classProperty,
                propertyAttribute,
                propertyAttributeProperty,
                propertyAttributeView,
                propertyAttributePropertyView,
                styleableProperty);
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
            const CesiumGltf::Schema& schema,
            const CesiumGltf::Class& classDefinition,
            const CesiumGltf::ClassProperty& classProperty,
            const CesiumGltf::PropertyTexture& propertyTexture,
            const CesiumGltf::PropertyTextureProperty& propertyTextureProperty,
            const CesiumGltf::PropertyTextureView& propertyTextureView,
            auto propertyTexturePropertyView) {
            using RawType = decltype(propertyTexturePropertyView.getRaw(0.0, 0.0));
            using TransformedType =
                typename std::decay_t<decltype(propertyTexturePropertyView.get(0.0, 0.0))>::value_type;
            constexpr bool IsArray = HAS_MEMBER(RawType, size());

            if constexpr (IsArray) {
                // Arrays are not supported
                return;
            } else {
                constexpr auto type = GetType<RawType, TransformedType>::Type;

                const auto textureInfo = GltfUtil::getPropertyTextureInfo(model, propertyTextureProperty);

                const auto styleableProperty = StyleablePropertyTexturePropertyInfo<type>{
                    textureInfo,
                    propertyTexturePropertyView.offset(),
                    propertyTexturePropertyView.scale(),
                    propertyTexturePropertyView.min(),
                    propertyTexturePropertyView.max(),
                    propertyTexturePropertyView.required(),
                    propertyTexturePropertyView.noData(),
                    propertyTexturePropertyView.defaultValue(),
                };

                callback(
                    propertyId,
                    schema,
                    classDefinition,
                    classProperty,
                    propertyTexture,
                    propertyTextureProperty,
                    propertyTextureView,
                    propertyTexturePropertyView,
                    styleableProperty);
            }
        });
}

std::vector<DataType>
getMdlPropertyAttributeTypes(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

std::vector<DataType>
getMdlPropertyTextureTypes(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

std::vector<const CesiumGltf::ImageCesium*>
getImagesReferencedByPropertyTextures(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

} // namespace cesium::omniverse::MetadataUtil
