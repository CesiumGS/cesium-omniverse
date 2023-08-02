#pragma once

#include "cesium/omniverse/FabricGeometry.h"
#include "cesium/omniverse/FabricGeometryDefinition.h"
#include "cesium/omniverse/ObjectPool.h"

namespace cesium::omniverse {

class FabricGeometryPool final : public ObjectPool<FabricGeometry> {
  public:
    FabricGeometryPool(
        int64_t poolId,
        const FabricGeometryDefinition& geometryDefinition,
        uint64_t initialCapacity,
        bool debugRandomColors,
        long stageId);

    [[nodiscard]] const FabricGeometryDefinition& getGeometryDefinition() const;

  protected:
    std::shared_ptr<FabricGeometry> createObject(uint64_t objectId) override;
    void setActive(std::shared_ptr<FabricGeometry> geometry, bool active) override;

  private:
    const int64_t _poolId;
    const FabricGeometryDefinition _geometryDefinition;
    const bool _debugRandomColors;
    const long _stageId;
};

} // namespace cesium::omniverse
