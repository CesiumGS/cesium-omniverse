#pragma once

#include <glm/glm.hpp>
#include <pxr/base/vt/array.h>
#include <pxr/usd/sdf/assetPath.h>

namespace CesiumGltf {
struct ImageCesium;
struct Material;
struct Model;
struct MeshPrimitive;
struct Texture;
} // namespace CesiumGltf

namespace cesium::omniverse::GltfUtil {

pxr::VtArray<pxr::GfVec3f> getPositions(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

std::optional<pxr::GfRange3d> getExtent(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

pxr::VtArray<int> getIndices(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const pxr::VtArray<pxr::GfVec3f>& positions);

pxr::VtArray<pxr::GfVec3f> getNormals(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const pxr::VtArray<pxr::GfVec3f>& positions,
    const pxr::VtArray<int>& indices,
    bool smoothNormals);

pxr::VtArray<pxr::GfVec2f> getTexcoords(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    uint64_t setIndex,
    const glm::fvec2& translation,
    const glm::fvec2& scale);

pxr::VtArray<int> getFaceVertexCounts(const pxr::VtArray<int>& indices);

pxr::GfVec3f getBaseColorFactor(const CesiumGltf::Material& material);
float getMetallicFactor(const CesiumGltf::Material& material);
float getRoughnessFactor(const CesiumGltf::Material& material);

pxr::GfVec3f getDefaultBaseColorFactor();
float getDefaultMetallicFactor();
float getDefaultRoughnessFactor();

std::optional<uint64_t> getBaseColorTextureIndex(const CesiumGltf::Model& model, const CesiumGltf::Material& material);

bool getDoubleSided(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

pxr::VtArray<pxr::GfVec2f> getImageryTexcoords(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    uint64_t setIndex,
    const glm::fvec2& translation,
    const glm::fvec2& scale);

const CesiumGltf::ImageCesium& getImageCesium(const CesiumGltf::Model& model, const CesiumGltf::Texture& texture);

bool hasNormals(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, bool smoothNormals);

bool hasTexcoords(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, uint64_t setIndex);

bool hasImageryTexcoords(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, uint64_t setIndex);

bool hasMaterial(const CesiumGltf::MeshPrimitive& primitive);

} // namespace cesium::omniverse::GltfUtil
