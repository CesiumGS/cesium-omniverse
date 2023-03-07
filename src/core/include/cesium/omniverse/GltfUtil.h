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

pxr::VtArray<pxr::GfVec3f>
getPrimitivePositions(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

std::optional<pxr::GfRange3d>
getPrimitiveExtent(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

pxr::VtArray<int> getPrimitiveIndices(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const pxr::VtArray<pxr::GfVec3f>& positions);

pxr::VtArray<pxr::GfVec3f> getPrimitiveNormals(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const pxr::VtArray<pxr::GfVec3f>& positions,
    const pxr::VtArray<int>& indices);

pxr::VtArray<pxr::GfVec2f>
getPrimitiveUvs(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, uint64_t setIndex);

pxr::VtArray<int> getPrimitiveFaceVertexCounts(const pxr::VtArray<int>& indices);

pxr::GfVec3f getBaseColorFactor(const CesiumGltf::Material& material);
float getMetallicFactor(const CesiumGltf::Material& material);
float getRoughnessFactor(const CesiumGltf::Material& material);

std::optional<uint64_t> getBaseColorTextureIndex(const CesiumGltf::Model& model, const CesiumGltf::Material& material);

bool getDoubleSided(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

pxr::VtArray<pxr::GfVec2f>
getImageryUvs(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, uint64_t setIndex);

const CesiumGltf::ImageCesium& getImageCesium(const CesiumGltf::Model& model, const CesiumGltf::Texture& texture);

} // namespace cesium::omniverse::GltfUtil
