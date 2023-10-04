#include "cesium/omniverse/FabricMaterialPool.h"

#include <spdlog/fmt/fmt.h>

namespace cesium::omniverse {

FabricMaterialPool::FabricMaterialPool(
    uint64_t poolId,
    const FabricMaterialDefinition& materialDefinition,
    uint64_t initialCapacity,
    const pxr::TfToken& defaultTextureAssetPathToken,
    long stageId,
    bool useTextureArray)
    : ObjectPool<FabricMaterial>()
    , _poolId(poolId)
    , _materialDefinition(materialDefinition)
    , _defaultTextureAssetPathToken(defaultTextureAssetPathToken)
    , _stageId(stageId)
    , _useTextureArray(useTextureArray) {
    setCapacity(initialCapacity);
}

const FabricMaterialDefinition& FabricMaterialPool::getMaterialDefinition() const {
    return _materialDefinition;
}

std::shared_ptr<FabricMaterial> FabricMaterialPool::createObject(uint64_t objectId) {
    const auto pathStr = fmt::format("/fabric_material_pool_{}_object_{}", _poolId, objectId);
    const auto path = omni::fabric::Path(pathStr.c_str());
    return std::make_shared<FabricMaterial>(
        path, _materialDefinition, _defaultTextureAssetPathToken, _stageId, _useTextureArray);
}

void FabricMaterialPool::setActive(std::shared_ptr<FabricMaterial> material, bool active) {
    material->setActive(active);
}
}; // namespace cesium::omniverse
