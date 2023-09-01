#pragma once

#include "cesium/omniverse/FabricTexture.h"
#include "cesium/omniverse/ObjectPool.h"

namespace cesium::omniverse {

class FabricTexturePool final : public ObjectPool<FabricTexture> {
  public:
    FabricTexturePool(uint64_t poolId, uint64_t initialCapacity);

  protected:
    std::shared_ptr<FabricTexture> createObject(uint64_t objectId) override;
    void setActive(std::shared_ptr<FabricTexture> texture, bool active) override;

  private:
    const uint64_t _poolId;
};

} // namespace cesium::omniverse
