#include "cesium/omniverse/FabricGeometryPool.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/FabricVertexAttributeDescriptor.h"
#include "cesium/omniverse/GltfUtil.h"

#include <spdlog/fmt/fmt.h>

namespace cesium::omniverse {

FabricGeometryPool::FabricGeometryPool(
    Context* pContext,
    int64_t poolId,
    const FabricGeometryDescriptor& geometryDescriptor,
    uint64_t initialCapacity)
    : ObjectPool<FabricGeometry>()
    , _pContext(pContext)
    , _poolId(poolId)
    , _geometryDescriptor(geometryDescriptor) {
    setCapacity(initialCapacity);
}

const FabricGeometryDescriptor& FabricGeometryPool::getGeometryDescriptor() const {
    return _geometryDescriptor;
}

int64_t FabricGeometryPool::getPoolId() const {
    return _poolId;
}

std::shared_ptr<FabricGeometry> FabricGeometryPool::createObject(uint64_t objectId) const {
    const auto contextId = _pContext->getContextId();
    const auto pathStr = fmt::format("/cesium_geometry_pool_{}_object_{}_context_{}", _poolId, objectId, contextId);
    const auto path = omni::fabric::Path(pathStr.c_str());
    return std::make_shared<FabricGeometry>(_pContext, path, _geometryDescriptor, _poolId);
}

void FabricGeometryPool::setActive(FabricGeometry* pGeometry, bool active) const {
    pGeometry->setActive(active);
}

}; // namespace cesium::omniverse
