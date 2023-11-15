#pragma once

#include "cesium/omniverse/DataType.h"
#include "cesium/omniverse/GltfUtil.h"

namespace cesium::omniverse::MetadataUtil {

template <typename Callback>
void forEachStyleablePropertyAttributeProperty(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    Callback&& callback) {

    GltfUtil::forEachPropertyAttributeProperty(
        model,
        primitive,
        [callback = std::forward<Callback>(callback)](
            const std::string& propertyId,
            const CesiumGltf::Schema& schema,
            const CesiumGltf::Class& classDefinition,
            const CesiumGltf::ClassProperty& classProperty,
            const CesiumGltf::PropertyAttributeView& propertyAttributeView,
            auto propertyAttributePropertyView) {
            const auto type = getClassPropertyType(schema, classProperty);

            if (type == DataType::UNKNOWN) {
                CESIUM_LOG_WARN("Unsupported property type. Property \"{}\" will be ignored.", propertyId);
                return;
            }

            const auto mdlType = getMdlPropertyType(type);

            if (mdlType == DataType::UNKNOWN) {
                CESIUM_LOG_WARN("Unsupported property type. Property \"{}\" will be ignored.", propertyId);
                return;
            }

            callback(
                propertyId,
                schema,
                classDefinition,
                classProperty,
                propertyAttributeView,
                propertyAttributePropertyView,
                type,
                mdlType);
        });
}

template <typename Callback>
void forEachStyleablePropertyTextureProperty(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    Callback&& callback) {

    GltfUtil::forEachPropertyTextureProperty(
        model,
        primitive,
        [callback = std::forward<Callback>(callback)](
            [[maybe_unused]] const std::string& propertyId,
            const CesiumGltf::Schema& schema,
            const CesiumGltf::Class& classDefinition,
            const CesiumGltf::ClassProperty& classProperty,
            const CesiumGltf::PropertyTextureView& propertyTextureView,
            [[maybe_unused]] auto propertyTexturePropertyView) {
            const auto type = getClassPropertyType(schema, classProperty);

            if (type == DataType::UNKNOWN) {
                CESIUM_LOG_WARN("Unsupported property type. Property \"{}\" will be ignored.", propertyId);
                return;
            }

            const auto mdlType = getMdlPropertyType(type);

            if (mdlType == DataType::UNKNOWN) {
                CESIUM_LOG_WARN("Unsupported property type. Property \"{}\" will be ignored.", propertyId);
                return;
            }

            callback(
                propertyId,
                schema,
                classDefinition,
                classProperty,
                propertyTextureView,
                propertyTexturePropertyView,
                type,
                mdlType);
        });
}

std::vector<DataType>
getMdlPropertyAttributeTypes(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
    std::vector<DataType> mdlTypes;

    forEachStyleablePropertyAttributeProperty(
        model,
        primitive,
        [&mdlTypes](
            [[maybe_unused]] const std::string& propertyId,
            [[maybe_unused]] const CesiumGltf::Schema& schema,
            [[maybe_unused]] const CesiumGltf::Class& classDefinition,
            [[maybe_unused]] const CesiumGltf::ClassProperty& classProperty,
            [[maybe_unused]] const CesiumGltf::PropertyAttributeView& propertyTextureView,
            [[maybe_unused]] auto propertyTexturePropertyView,
            [[maybe_unused]] DataType type,
            DataType mdlType) { mdlTypes.push_back(mdlType); });

    std::sort(mdlTypes.begin(), mdlTypes.end());

    return mdlTypes;
}

std::vector<DataType>
getMdlPropertyTextureTypes(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
    std::vector<DataType> mdlTypes;

    forEachStyleablePropertyTextureProperty(
        model,
        primitive,
        [&mdlTypes](
            [[maybe_unused]] const std::string& propertyId,
            [[maybe_unused]] const CesiumGltf::Schema& schema,
            [[maybe_unused]] const CesiumGltf::Class& classDefinition,
            [[maybe_unused]] const CesiumGltf::ClassProperty& classProperty,
            [[maybe_unused]] const CesiumGltf::PropertyTextureView& propertyTextureView,
            [[maybe_unused]] auto propertyTexturePropertyView,
            [[maybe_unused]] DataType type,
            DataType mdlType) { mdlTypes.push_back(mdlType); });

    std::sort(mdlTypes.begin(), mdlTypes.end());

    return mdlTypes;
}

} // namespace cesium::omniverse::MetadataUtil
