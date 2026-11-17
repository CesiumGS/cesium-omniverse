#include "cesium/omniverse/FabricTexturePool.h"

#include "cesium/omniverse/Context.h"

#include <fmt/format.h>

namespace cesium::omniverse {

FabricTexturePool::FabricTexturePool(Context* pContext, int64_t poolId, uint64_t initialCapacity)
    : ObjectPool<FabricTexture>()
    , _pContext(pContext)
    , _poolId(poolId) {
    setCapacity(initialCapacity);
}

int64_t FabricTexturePool::getPoolId() const {
    return _poolId;
}

std::shared_ptr<FabricTexture> FabricTexturePool::createObject(uint64_t objectId) const {
    // Creating a Fabric prim with the same path as a previously destroyed prim causes a crash.
    // The contextId is time-tagged and ensures that Fabric prim paths are unique even across extension reloads.
    const auto contextId = _pContext->getContextId();
    const auto name = fmt::format("/cesium_texture_pool_{}_object_{}_context_{}", _poolId, objectId, contextId);
    return std::make_shared<FabricTexture>(_pContext, name, _poolId);
}

void FabricTexturePool::setActive(FabricTexture* pTexture, bool active) const {
    pTexture->setActive(active);
}

}; // namespace cesium::omniverse
