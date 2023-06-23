#include "cesium/omniverse/OmniMesh.h"

#include "cesium/omniverse/UsdUtil.h"

namespace cesium::omniverse {
OmniMesh::OmniMesh(std::shared_ptr<OmniGeometry> geometry, std::shared_ptr<OmniMaterial> material)
    : _geometry(geometry)
    , _material(material) {}

std::shared_ptr<OmniGeometry> OmniMesh::getGeometry() const {
    return _geometry;
}
std::shared_ptr<OmniMaterial> OmniMesh::getMaterial() const {
    return _material;
}

void OmniMesh::setVisibility(bool visible) const {
    _geometry->setVisibility(visible);
}

void OmniMesh::setTile(
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
        imageryTexcoordTranslation,
        imageryTexcoordScale,
        imageryTexcoordSetIndex);

    if (material != nullptr) {
        material->setTile(tilesetId, tileId, model, primitive, imagery);
        geometry->assignMaterial(material);
    }
}

} // namespace cesium::omniverse
