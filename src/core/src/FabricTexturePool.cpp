#include "cesium/omniverse/FabricTexturePool.h"

#include <spdlog/fmt/fmt.h>

namespace cesium::omniverse {

FabricTexturePool::FabricTexturePool(int64_t poolId, uint64_t initialCapacity)
    : ObjectPool<FabricTexture>()
    , _poolId(poolId) {
    setCapacity(initialCapacity);
}

std::shared_ptr<FabricTexture> FabricTexturePool::createObject(uint64_t objectId) {
    const auto name = fmt::format("/fabric_texture_pool_{}_object_{}", _poolId, objectId);
    return std::make_shared<FabricTexture>(name);
}

void FabricTexturePool::setActive(std::shared_ptr<FabricTexture> texture, bool active) {
    texture->setActive(active);
}

}; // namespace cesium::omniverse
