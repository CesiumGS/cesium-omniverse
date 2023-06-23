#pragma once

#include "cesium/omniverse/ObjectPool.h"
#include "cesium/omniverse/OmniGeometry.h"
#include "cesium/omniverse/OmniGeometryDefinition.h"
#include "cesium/omniverse/OmniSceneDelegate.h"

namespace cesium::omniverse {

class OmniGeometryPool final : public ObjectPool<OmniGeometry> {
  public:
    OmniGeometryPool(
        int64_t poolId,
        const OmniGeometryDefinition& geometryDefinition,
        uint64_t initialCapacity,
        bool debugRandomColors,
        OmniSceneDelegate sceneDelegate);

    const OmniGeometryDefinition& getGeometryDefinition() const;

  protected:
    std::shared_ptr<OmniGeometry> createObject(uint64_t objectId) override;
    void setActive(std::shared_ptr<OmniGeometry> geometry, bool active) override;

  private:
    const int64_t _poolId;
    const OmniGeometryDefinition _geometryDefinition;
    const bool _debugRandomColors;
    const OmniSceneDelegate _sceneDelegate;
};

} // namespace cesium::omniverse
