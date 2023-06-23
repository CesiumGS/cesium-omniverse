#include "cesium/omniverse/OmniMaterialPool.h"

#include "cesium/omniverse/FabricMaterial.h"

#include <spdlog/fmt/fmt.h>

namespace cesium::omniverse {

OmniMaterialPool::OmniMaterialPool(
    int64_t poolId,
    const OmniMaterialDefinition& materialDefinition,
    uint64_t initialCapacity,
    OmniSceneDelegate sceneDelegate)
    : ObjectPool<OmniMaterial>()
    , _poolId(poolId)
    , _materialDefinition(materialDefinition)
    , _sceneDelegate(sceneDelegate) {
    setCapacity(initialCapacity);
}

const OmniMaterialDefinition& OmniMaterialPool::getMaterialDefinition() const {
    return _materialDefinition;
}

std::shared_ptr<OmniMaterial> OmniMaterialPool::createObject(uint64_t objectId) {
    const auto path = pxr::SdfPath(fmt::format("/omni_material_pool_{}_object_{}", _poolId, objectId));

    switch (_sceneDelegate) {
        case OmniSceneDelegate::FABRIC: {
            return std::make_shared<FabricMaterial>(path, _materialDefinition);
        }
        case OmniSceneDelegate::USD: {
            return std::make_shared<FabricMaterial>(path, _materialDefinition);
        }
        default: {
            assert(false);
            return nullptr;
        }
    }
}

void OmniMaterialPool::setActive(std::shared_ptr<OmniMaterial> material, bool active) {
    material->setActive(active);
}

}; // namespace cesium::omniverse
