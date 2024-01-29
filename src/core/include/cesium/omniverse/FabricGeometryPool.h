#pragma once

#include "cesium/omniverse/FabricGeometry.h"
#include "cesium/omniverse/FabricGeometryDescriptor.h"
#include "cesium/omniverse/ObjectPool.h"

namespace cesium::omniverse {

class FabricGeometryPool final : public ObjectPool<FabricGeometry> {
  public:
    FabricGeometryPool(
        Context* pContext,
        int64_t poolId,
        const FabricGeometryDescriptor& geometryDescriptor,
        uint64_t initialCapacity);
    ~FabricGeometryPool() override = default;
    FabricGeometryPool(const FabricGeometryPool&) = delete;
    FabricGeometryPool& operator=(const FabricGeometryPool&) = delete;
    FabricGeometryPool(FabricGeometryPool&&) noexcept = default;
    FabricGeometryPool& operator=(FabricGeometryPool&&) noexcept = default;

    [[nodiscard]] const FabricGeometryDescriptor& getGeometryDescriptor() const;
    [[nodiscard]] int64_t getPoolId() const;

  protected:
    [[nodiscard]] std::shared_ptr<FabricGeometry> createObject(uint64_t objectId) const override;
    void setActive(FabricGeometry* pGeometry, bool active) const override;

  private:
    Context* _pContext;
    int64_t _poolId;
    FabricGeometryDescriptor _geometryDescriptor;
};

} // namespace cesium::omniverse
