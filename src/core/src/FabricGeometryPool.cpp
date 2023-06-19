#include "cesium/omniverse/FabricGeometryPool.h"

#include <spdlog/fmt/fmt.h>

namespace cesium::omniverse {

FabricGeometryPool::FabricGeometryPool(
    int64_t poolId,
    const FabricGeometryDefinition& geometryDefinition,
    uint64_t initialCapacity,
    bool debugRandomColors)
    : ObjectPool<FabricGeometry>()
    , _poolId(poolId)
    , _geometryDefinition(geometryDefinition)
    , _debugRandomColors(debugRandomColors) {
    setCapacity(initialCapacity);
}

const FabricGeometryDefinition& FabricGeometryPool::getGeometryDefinition() const {
    return _geometryDefinition;
}

std::shared_ptr<FabricGeometry> FabricGeometryPool::createObject(uint64_t objectId) {
    const auto path = pxr::SdfPath(fmt::format("/fabric_geometry_pool_{}_object_{}", _poolId, objectId));
    return std::make_shared<FabricGeometry>(path, _geometryDefinition, _debugRandomColors);
}

void FabricGeometryPool::setActive(std::shared_ptr<FabricGeometry> geometry, bool active) {
    geometry->setActive(active);
}

}; // namespace cesium::omniverse
