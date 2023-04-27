#include "cesium/omniverse/FabricMaterialPool.h"

#include <spdlog/fmt/fmt.h>

namespace cesium::omniverse {

FabricMaterialPool::FabricMaterialPool(int64_t poolId, const FabricMaterialDefinition& materialDefinition)
    : ObjectPool<FabricMaterial>()
    , _poolId(poolId)
    , _materialDefinition(materialDefinition) {

    // Material creation is expensive and often stalls the main thread so create as many as we can upfront.
    // The number below is roughly the number of tiles cesium-native keeps in memory before hitting the
    // default maximum memory usage for Cesium World Terrain.
    setCapacity(2000);
}

const FabricMaterialDefinition& FabricMaterialPool::getMaterialDefinition() const {
    return _materialDefinition;
}

std::shared_ptr<FabricMaterial> FabricMaterialPool::createObject(uint64_t objectId) {
    const auto path = pxr::SdfPath(fmt::format("/fabric_material_pool_{}_object_{}", _poolId, objectId));
    return std::make_shared<FabricMaterial>(path, _materialDefinition);
}

void FabricMaterialPool::setActive(std::shared_ptr<FabricMaterial> material, bool active) {
    material->setActive(active);
}

}; // namespace cesium::omniverse
