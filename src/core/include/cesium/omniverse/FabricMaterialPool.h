#pragma once

#include "cesium/omniverse/FabricMaterial.h"
#include "cesium/omniverse/FabricMaterialDefinition.h"
#include "cesium/omniverse/ObjectPool.h"

#include <pxr/usd/sdf/assetPath.h>

namespace cesium::omniverse {

class FabricMaterialPool final : public ObjectPool<FabricMaterial> {
  public:
    FabricMaterialPool(
        uint64_t poolId,
        const FabricMaterialDefinition& materialDefinition,
        uint64_t initialCapacity,
        const pxr::TfToken& defaultTextureAssetPathToken,
        long stageId,
        bool useTextureArray);

    [[nodiscard]] const FabricMaterialDefinition& getMaterialDefinition() const;

  protected:
    std::shared_ptr<FabricMaterial> createObject(uint64_t objectId) override;
    void setActive(std::shared_ptr<FabricMaterial> material, bool active) override;

  private:
    const uint64_t _poolId;
    const FabricMaterialDefinition _materialDefinition;
    const pxr::TfToken _defaultTextureAssetPathToken;
    const long _stageId;
    const bool _useTextureArray;
};

} // namespace cesium::omniverse
