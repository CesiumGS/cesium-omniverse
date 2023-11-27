#pragma once

#include "cesium/omniverse/DataType.h"
#include "cesium/omniverse/GltfUtil.h"
#include "cesium/omniverse/LoggerSink.h"

#include <CesiumGltf/ExtensionMeshPrimitiveExtStructuralMetadata.h>
#include <CesiumGltf/ExtensionModelExtStructuralMetadata.h>
#include <CesiumGltf/PropertyAttribute.h>
#include <CesiumGltf/PropertyAttributeView.h>

namespace cesium::omniverse::MetadataUtil {

template <DataType T> struct StyleablePropertyInfo {
    std::optional<GetTransformedType<T>> offset;
    std::optional<GetTransformedType<T>> scale;
    std::optional<GetTransformedType<T>> min;
    std::optional<GetTransformedType<T>> max;
    bool required;
    std::optional<GetRawType<T>> noData;
    std::optional<GetTransformedType<T>> defaultValue;
};

template <DataType T> struct StyleablePropertyAttributePropertyInfo {
    static constexpr auto Type = T;
    std::string attribute;
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

                const auto propertyInfo = StyleablePropertyInfo<type>{
                    propertyAttributePropertyView.offset(),
                    propertyAttributePropertyView.scale(),
                    propertyAttributePropertyView.min(),
                    propertyAttributePropertyView.max(),
                    propertyAttributePropertyView.required(),
                    propertyAttributePropertyView.noData(),
                    propertyAttributePropertyView.defaultValue(),
                };

                const auto styleableProperty = StyleablePropertyAttributePropertyInfo<type>{
                    attribute,
                    propertyInfo,
                };

                callback(propertyId, propertyAttributePropertyView, styleableProperty);
            }
        });
}

std::vector<MdlInternalPropertyType> getMdlInternalPropertyAttributePropertyTypes(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive);

} // namespace cesium::omniverse::MetadataUtil
