#pragma once

#include "cesium/omniverse/GltfAccessors.h"
#include "cesium/omniverse/VertexAttributeType.h"

#include <CesiumGltf/Accessor.h>
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

struct ImageryLayersInfo {
    uint64_t imageryLayerCount;
};

FeatureIdType getFeatureIdType(const FeatureId& featureId);
std::vector<FeatureIdType> getFeatureIdTypes(const FeaturesInfo& featuresInfo);
std::vector<uint64_t> getSetIndexMapping(const FeaturesInfo& featuresInfo, FeatureIdType type);
bool hasFeatureIdType(const FeaturesInfo& featuresInfo, FeatureIdType type);

struct VertexAttributeInfo {
    VertexAttributeType type;
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

const MaterialInfo& getDefaultMaterialInfo();
const TextureInfo& getDefaultTextureInfo();

bool hasNormals(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, bool smoothNormals);
bool hasTexcoords(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, uint64_t setIndex);
bool hasImageryTexcoords(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, uint64_t setIndex);
bool hasVertexColors(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, uint64_t setIndex);
bool hasMaterial(const CesiumGltf::MeshPrimitive& primitive);

std::vector<uint64_t> getTexcoordSetIndexes(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);
std::vector<uint64_t>
getImageryTexcoordSetIndexes(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

template <VertexAttributeType T>
VertexAttributeAccessor<T> getVertexAttributeValues(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const std::string& attributeName) {

    const auto attribute = primitive.attributes.find(attributeName);
    if (attribute == primitive.attributes.end()) {
        return {};
    }

    auto pAccessor = model.getSafe<CesiumGltf::Accessor>(&model.accessors, attribute->second);
    if (!pAccessor) {
        return {};
    }

    auto view = CesiumGltf::AccessorView<GetRawType<T>>(model, *pAccessor);

    if (view.status() != CesiumGltf::AccessorViewStatus::Valid) {
        return {};
    }

    return VertexAttributeAccessor<T>(view);
}

} // namespace cesium::omniverse::GltfUtil
