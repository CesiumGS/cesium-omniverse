#pragma once

#include "cesium/omniverse/DataType.h"
#include "cesium/omniverse/DataTypeUtil.h"
#include "cesium/omniverse/VertexAttributeAccessors.h"

#include <CesiumGltf/Accessor.h>
#include <glm/fwd.hpp>

#include <set>

namespace CesiumGltf {
struct Material;
struct Model;
struct MeshPrimitive;
struct Texture;
struct PropertyTextureProperty;
} // namespace CesiumGltf

namespace CesiumImage {
struct ImageAsset;
}

namespace cesium::omniverse {
struct FeaturesInfo;
struct MaterialInfo;
struct TextureInfo;
} // namespace cesium::omniverse

namespace cesium::omniverse::GltfUtil {

std::optional<DataType> getVertexAttributeTypeFromGltf(const CesiumGltf::Accessor& accessor);

std::pair<std::string, uint64_t> parseAttributeName(const std::string& attributeName);

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

TexcoordsAccessor getRasterOverlayTexcoords(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    uint64_t setIndex);

VertexColorsAccessor
getVertexColors(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, uint64_t setIndex);

VertexIdsAccessor getVertexIds(const PositionsAccessor& positionsAccessor);

const CesiumImage::ImageAsset*
getBaseColorTextureImage(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

const CesiumImage::ImageAsset* getFeatureIdTextureImage(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    uint64_t featureIdSetIndex);

MaterialInfo getMaterialInfo(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

FeaturesInfo getFeaturesInfo(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

const MaterialInfo& getDefaultMaterialInfo();
const TextureInfo& getDefaultTextureInfo();
TextureInfo getPropertyTexturePropertyInfo(
    const CesiumGltf::Model& model,
    const CesiumGltf::PropertyTextureProperty& propertyTextureProperty);

bool hasNormals(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, bool smoothNormals);
bool hasTexcoords(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, uint64_t setIndex);
bool hasRasterOverlayTexcoords(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    uint64_t setIndex);
bool hasVertexColors(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, uint64_t setIndex);
bool hasMaterial(const CesiumGltf::MeshPrimitive& primitive);

std::vector<uint64_t> getTexcoordSetIndexes(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);
std::vector<uint64_t>
getRasterOverlayTexcoordSetIndexes(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

CesiumImage::Ktx2TranscodeTargets getKtx2TranscodeTargets();

template <DataType T>
VertexAttributeAccessor<T> getVertexAttributeValues(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const std::string& attributeName) {

    const auto it = primitive.attributes.find(attributeName);
    if (it == primitive.attributes.end()) {
        return {};
    }

    const auto pAccessor = model.getSafe(&model.accessors, it->second);
    if (!pAccessor) {
        return {};
    }

    const auto view = CesiumGltf::AccessorView<DataTypeUtil::GetNativeType<T>>(model, *pAccessor);

    if (view.status() != CesiumGltf::AccessorViewStatus::Valid) {
        return {};
    }

    return VertexAttributeAccessor<T>(view);
}

} // namespace cesium::omniverse::GltfUtil
