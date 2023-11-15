#pragma once

#include "cesium/omniverse/DataType.h"
#include "cesium/omniverse/GltfAccessors.h"
#include "cesium/omniverse/LoggerSink.h"

#include <CesiumGltf/Accessor.h>
#include <CesiumGltf/ExtensionMeshPrimitiveExtStructuralMetadata.h>
#include <CesiumGltf/ExtensionModelExtStructuralMetadata.h>
#include <CesiumGltf/PropertyAttributePropertyView.h>
#include <CesiumGltf/PropertyAttributeView.h>
#include <CesiumGltf/PropertyTexturePropertyView.h>
#include <CesiumGltf/PropertyTextureView.h>
#include <glm/glm.hpp>
#include <omni/fabric/core/FabricTypes.h>

#include <set>
#include <variant>

namespace CesiumGltf {
struct ImageCesium;
struct Material;
struct Model;
struct MeshPrimitive;
struct Texture;
} // namespace CesiumGltf

namespace cesium::omniverse {

enum class AlphaMode : int {
    OPAQUE = 0,
    MASK = 1,
    BLEND = 2,
};

struct TextureInfo {
    glm::dvec2 offset;
    double rotation;
    glm::dvec2 scale;
    uint64_t setIndex;
    int32_t wrapS;
    int32_t wrapT;
    bool flipVertical;
    std::vector<uint8_t> channels;

    // Make sure to update this function when adding new fields to the struct
    bool operator==(const TextureInfo& other) const;
};

struct MaterialInfo {
    double alphaCutoff;
    AlphaMode alphaMode;
    double baseAlpha;
    glm::dvec3 baseColorFactor;
    glm::dvec3 emissiveFactor;
    double metallicFactor;
    double roughnessFactor;
    bool doubleSided;
    bool hasVertexColors;
    std::optional<TextureInfo> baseColorTexture;

    // Make sure to update this function when adding new fields to the struct
    bool operator==(const MaterialInfo& other) const;
};

enum class FeatureIdType {
    INDEX,
    ATTRIBUTE,
    TEXTURE,
};

struct FeatureId {
    std::optional<uint64_t> nullFeatureId;
    uint64_t featureCount;
    std::variant<std::monostate, uint64_t, TextureInfo> featureIdStorage;
};

struct FeaturesInfo {
    std::vector<FeatureId> featureIds;
};

FeatureIdType getFeatureIdType(const FeatureId& featureId);
std::vector<FeatureIdType> getFeatureIdTypes(const FeaturesInfo& featuresInfo);
std::vector<uint64_t> getSetIndexMapping(const FeaturesInfo& featuresInfo, FeatureIdType type);
bool hasFeatureIdType(const FeaturesInfo& featuresInfo, FeatureIdType type);

struct VertexAttributeInfo {
    DataType type;
    omni::fabric::Token fabricAttributeName;
    std::string gltfAttributeName;

    // Make sure to update this function when adding new fields to the struct
    bool operator==(const VertexAttributeInfo& other) const;
    bool operator<(const VertexAttributeInfo& other) const;
};

} // namespace cesium::omniverse

namespace cesium::omniverse::GltfUtil {

PositionsAccessor getPositions(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

std::optional<std::array<glm::dvec3, 2>>
getExtent(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

IndicesAccessor getIndices(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const PositionsAccessor& positions);

FaceVertexCountsAccessor getFaceVertexCounts(const IndicesAccessor& indices);

NormalsAccessor getNormals(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const PositionsAccessor& positionsAccessor,
    const IndicesAccessor& indices,
    bool smoothNormals);

TexcoordsAccessor
getTexcoords(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, uint64_t setIndex);

TexcoordsAccessor
getImageryTexcoords(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, uint64_t setIndex);

VertexColorsAccessor
getVertexColors(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, uint64_t setIndex);

VertexIdsAccessor getVertexIds(const PositionsAccessor& positionsAccessor);

template <DataType T>
VertexAttributeAccessor<T> getVertexAttributeValues(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const std::string& attributeName) {

    const auto attribute = primitive.attributes.find(attributeName);
    if (attribute == primitive.attributes.end()) {
        return {};
    }

    auto accessor = model.getSafe<CesiumGltf::Accessor>(&model.accessors, attribute->second);
    if (!accessor) {
        return {};
    }

    auto view = CesiumGltf::AccessorView<GetNativeType<T>>(model, *accessor);

    if (view.status() != CesiumGltf::AccessorViewStatus::Valid) {
        return {};
    }

    return VertexAttributeAccessor<T>(view);
}

const CesiumGltf::ImageCesium*
getBaseColorTextureImage(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

const CesiumGltf::ImageCesium* getFeatureIdTextureImage(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    uint64_t featureIdSetIndex);

MaterialInfo getMaterialInfo(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

FeaturesInfo getFeaturesInfo(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

std::set<VertexAttributeInfo>
getCustomVertexAttributes(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

MaterialInfo getDefaultMaterialInfo();
TextureInfo getDefaultTextureInfo();

bool hasNormals(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, bool smoothNormals);
bool hasTexcoords(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, uint64_t setIndex);
bool hasImageryTexcoords(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, uint64_t setIndex);
bool hasVertexColors(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, uint64_t setIndex);
bool hasMaterial(const CesiumGltf::MeshPrimitive& primitive);

std::vector<uint64_t> getTexcoordSetIndexes(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);
std::vector<uint64_t>
getImageryTexcoordSetIndexes(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

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
            [callback = std::forward<Callback>(callback), &propertyAttributeView, &pStructuralMetadataModel](
                [[maybe_unused]] const std::string& propertyId, auto propertyAttributePropertyView) {
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

                callback(propertyId, schema.value(), propertyAttributeView, propertyAttributePropertyView);
            });
    }
}

template <typename Callback>
void forEachPropertyTexture(
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
            [callback = std::forward<Callback>(callback), &propertyTextureView, &pStructuralMetadataModel](
                [[maybe_unused]] const std::string& propertyId, auto propertyTexturePropertyView) {
                if (propertyTexturePropertyView.status() != CesiumGltf::PropertyTexturePropertyViewStatus::Valid) {
                    CESIUM_LOG_WARN(
                        "Property \"{}\" is invalid and will be ignored. Status code: {}",
                        propertyId,
                        static_cast<int>(propertyTexturePropertyView.status()));
                    return;
                }

                const auto& schema = pStructuralMetadataModel->schema;
                if (!schema.has_value()) {
                    CESIUM_LOG_WARN("No schema found. Property will be ignored. Status code: {}", propertyId);
                    return;
                }

                callback(propertyId, schema.value(), propertyTextureView, propertyTexturePropertyView);
            });
    }
}

} // namespace cesium::omniverse::GltfUtil
