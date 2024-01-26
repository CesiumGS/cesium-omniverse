#pragma once

#include "cesium/omniverse/DataType.h"
#include "cesium/omniverse/FabricVertexAttributeAccessors.h"

#include <CesiumGltf/Accessor.h>
#include <glm/fwd.hpp>

#include <set>

namespace CesiumGltf {
struct ImageCesium;
struct Material;
struct Model;
struct MeshPrimitive;
struct Texture;
struct PropertyTextureProperty;
} // namespace CesiumGltf

namespace cesium::omniverse {
struct FabricFeaturesInfo;
struct FabricMaterialInfo;
struct FabricTextureInfo;
struct FabricVertexAttributeDescriptor;
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
getRasterOverlayTexcoords(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, uint64_t setIndex);

VertexColorsAccessor
getVertexColors(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, uint64_t setIndex);

VertexIdsAccessor getVertexIds(const PositionsAccessor& positionsAccessor);

const CesiumGltf::ImageCesium*
getBaseColorTextureImage(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

const CesiumGltf::ImageCesium* getFeatureIdTextureImage(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    uint64_t featureIdSetIndex);

FabricMaterialInfo getMaterialInfo(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

FabricFeaturesInfo getFeaturesInfo(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

std::set<FabricVertexAttributeDescriptor>
getCustomVertexAttributes(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

const FabricMaterialInfo& getDefaultMaterialInfo();
const FabricTextureInfo& getDefaultTextureInfo();
FabricTextureInfo getPropertyTexturePropertyInfo(
    const CesiumGltf::Model& model,
    const CesiumGltf::PropertyTextureProperty& propertyTextureProperty);

bool hasNormals(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, bool smoothNormals);
bool hasTexcoords(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, uint64_t setIndex);
bool hasRasterOverlayTexcoords(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, uint64_t setIndex);
bool hasVertexColors(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, uint64_t setIndex);
bool hasMaterial(const CesiumGltf::MeshPrimitive& primitive);

std::vector<uint64_t> getTexcoordSetIndexes(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);
std::vector<uint64_t>
getRasterOverlayTexcoordSetIndexes(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

CesiumGltf::Ktx2TranscodeTargets getKtx2TranscodeTargets();

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
