#include "cesium/omniverse/FabricMesh.h"

#include "cesium/omniverse/UsdUtil.h"

namespace cesium::omniverse {
FabricMesh::FabricMesh(std::shared_ptr<FabricGeometry> geometry, std::shared_ptr<FabricMaterial> material)
    : _geometry(std::move(geometry))
    , _material(std::move(material)) {}

std::shared_ptr<FabricGeometry> FabricMesh::getGeometry() const {
    return _geometry;
}
std::shared_ptr<FabricMaterial> FabricMesh::getMaterial() const {
    return _material;
}

void FabricMesh::setVisibility(bool visible) const {
    _geometry->setVisibility(visible);
}

void FabricMesh::setTile(
    int64_t tilesetId,
    int64_t tileId,
    const glm::dmat4& ecefToUsdTransform,
    const glm::dmat4& gltfToEcefTransform,
    const glm::dmat4& nodeTransform,
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    bool smoothNormals,
    const CesiumGltf::ImageCesium* imagery,
    const glm::dvec2& imageryTexcoordTranslation,
    const glm::dvec2& imageryTexcoordScale,
    uint64_t imageryTexcoordSetIndex) {

    const auto geometry = getGeometry();
    const auto material = getMaterial();

    assert(geometry != nullptr);

    const auto hasImagery = imagery != nullptr;

    geometry->setTile(
        tilesetId,
        tileId,
        ecefToUsdTransform,
        gltfToEcefTransform,
        nodeTransform,
        model,
        primitive,
        smoothNormals,
        hasImagery,
        imageryTexcoordSetIndex);

    if (material != nullptr) {
        material->setTile(
            tilesetId, tileId, model, primitive, imagery, imageryTexcoordTranslation, imageryTexcoordScale);
        geometry->assignMaterial(material);
    }
}

} // namespace cesium::omniverse
