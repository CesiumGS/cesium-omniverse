#include "cesium/omniverse/FabricMesh.h"

#include "cesium/omniverse/UsdUtil.h"

namespace cesium::omniverse {
FabricMesh::FabricMesh(std::shared_ptr<FabricGeometry> geometry, std::shared_ptr<FabricMaterial> material)
    : _geometry(geometry)
    , _material(material) {}

std::shared_ptr<FabricGeometry> FabricMesh::getGeometry() const {
    return _geometry;
}
std::shared_ptr<FabricMaterial> FabricMesh::getMaterial() const {
    return _material;
}

void FabricMesh::setVisibility(bool visible) const {
    _geometry->setVisibility(visible);
}

} // namespace cesium::omniverse
