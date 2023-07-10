#pragma once

#include "cesium/omniverse/GltfAccessors.h"

#include <glm/glm.hpp>

namespace CesiumGltf {
struct ImageCesium;
struct Material;
struct Model;
struct MeshPrimitive;
struct Texture;
} // namespace CesiumGltf

namespace cesium::omniverse {
struct TextureInfo {
    glm::dvec2 offset;
    double rotation;
    glm::dvec2 scale;
    uint64_t setIndex;
    int32_t wrapS;
    int32_t wrapT;
    bool flipVertical;
};

struct MaterialInfo {
    double alphaCutoff;
    int32_t alphaMode;
    double baseAlpha;
    glm::dvec3 baseColorFactor;
    glm::dvec3 emissiveFactor;
    double metallicFactor;
    double roughnessFactor;
    bool doubleSided;
    bool hasVertexColors;
    std::optional<TextureInfo> baseColorTexture;
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

const CesiumGltf::ImageCesium*
getBaseColorTextureImage(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

MaterialInfo getMaterialInfo(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

MaterialInfo getDefaultMaterialInfo();
TextureInfo getDefaultTextureInfo();

bool hasNormals(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, bool smoothNormals);
bool hasTexcoords(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, uint64_t setIndex);
bool hasImageryTexcoords(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, uint64_t setIndex);
bool hasVertexColors(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, uint64_t setIndex);
bool hasMaterial(const CesiumGltf::MeshPrimitive& primitive);

} // namespace cesium::omniverse::GltfUtil
