#pragma once

#include "cesium/omniverse/FabricTexture.h"
#include "cesium/omniverse/ObjectPool.h"

namespace cesium::omniverse {

class FabricTexturePool final : public ObjectPool<FabricTexture> {
  public:
    FabricTexturePool(Context* pContext, int64_t poolId, uint64_t initialCapacity);
    ~FabricTexturePool() override = default;
    FabricTexturePool(const FabricTexturePool&) = delete;
    FabricTexturePool& operator=(const FabricTexturePool&) = delete;
    FabricTexturePool(FabricTexturePool&&) noexcept = default;
    FabricTexturePool& operator=(FabricTexturePool&&) noexcept = default;

    [[nodiscard]] int64_t getPoolId() const;

  protected:
    [[nodiscard]] std::shared_ptr<FabricTexture> createObject(uint64_t objectId) const override;
    void setActive(FabricTexture* pTexture, bool active) const override;

  private:
    Context* _pContext;
    int64_t _poolId;
};

} // namespace cesium::omniverse
