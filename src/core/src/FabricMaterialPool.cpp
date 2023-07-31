#include "cesium/omniverse/FabricMaterialPool.h"

#include <spdlog/fmt/fmt.h>

namespace cesium::omniverse {

FabricMaterialPool::FabricMaterialPool(
    int64_t poolId,
    const FabricMaterialDefinition& materialDefinition,
    uint64_t initialCapacity,
    const pxr::TfToken& defaultTextureAssetPathToken)
    : ObjectPool<FabricMaterial>()
    , _poolId(poolId)
    , _materialDefinition(materialDefinition)
    , _defaultTextureAssetPathToken(defaultTextureAssetPathToken) {
    setCapacity(initialCapacity);
}

const FabricMaterialDefinition& FabricMaterialPool::getMaterialDefinition() const {
    return _materialDefinition;
}

std::shared_ptr<FabricMaterial> FabricMaterialPool::createObject(uint64_t objectId) {
    const auto path = pxr::SdfPath(fmt::format("/fabric_material_pool_{}_object_{}", _poolId, objectId));
    return std::make_shared<FabricMaterial>(path, _materialDefinition, _defaultTextureAssetPathToken);
}

void FabricMaterialPool::setActive(std::shared_ptr<FabricMaterial> material, bool active) {
    material->setActive(active);
}

}; // namespace cesium::omniverse
