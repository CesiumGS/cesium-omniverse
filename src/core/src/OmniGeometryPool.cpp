#include "cesium/omniverse/OmniGeometryPool.h"

#include "cesium/omniverse/FabricGeometry.h"

#include <spdlog/fmt/fmt.h>

namespace cesium::omniverse {

OmniGeometryPool::OmniGeometryPool(
    int64_t poolId,
    const OmniGeometryDefinition& geometryDefinition,
    uint64_t initialCapacity,
    bool debugRandomColors)
    : ObjectPool<OmniGeometry>()
    , _poolId(poolId)
    , _geometryDefinition(geometryDefinition)
    , _debugRandomColors(debugRandomColors) {
    setCapacity(initialCapacity);
}

const OmniGeometryDefinition& OmniGeometryPool::getGeometryDefinition() const {
    return _geometryDefinition;
}

std::shared_ptr<OmniGeometry> OmniGeometryPool::createObject(uint64_t objectId) {
    const auto path = pxr::SdfPath(fmt::format("/omni_geometry_pool_{}_object_{}", _poolId, objectId));
    return std::make_shared<FabricGeometry>(path, _geometryDefinition, _debugRandomColors);
}

void OmniGeometryPool::setActive(std::shared_ptr<OmniGeometry> geometry, bool active) {
    geometry->setActive(active);
}

}; // namespace cesium::omniverse
