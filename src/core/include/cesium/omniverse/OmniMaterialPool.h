#pragma once

#include "cesium/omniverse/ObjectPool.h"
#include "cesium/omniverse/OmniMaterial.h"
#include "cesium/omniverse/OmniMaterialDefinition.h"
#include "cesium/omniverse/OmniSceneDelegate.h"

namespace cesium::omniverse {

class OmniMaterialPool final : public ObjectPool<OmniMaterial> {
  public:
    OmniMaterialPool(
        int64_t poolId,
        const OmniMaterialDefinition& materialDefinition,
        uint64_t initialCapacity,
        OmniSceneDelegate sceneDelegate);

    const OmniMaterialDefinition& getMaterialDefinition() const;

  protected:
    std::shared_ptr<OmniMaterial> createObject(uint64_t objectId) override;
    void setActive(std::shared_ptr<OmniMaterial> material, bool active) override;

  private:
    const int64_t _poolId;
    const OmniMaterialDefinition _materialDefinition;
    const OmniSceneDelegate _sceneDelegate;
};

} // namespace cesium::omniverse
