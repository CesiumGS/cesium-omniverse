#pragma once

#include "cesium/omniverse/OmniImagery.h"

#include <CesiumRasterOverlays/RasterizedPolygonsOverlay.h>
#include <CesiumUtility/IntrusivePointer.h>

namespace cesium::omniverse {

class OmniPolygonImagery final : public OmniImagery {
  public:
    OmniPolygonImagery(Context* pContext, const PXR_NS::SdfPath& path);
    ~OmniPolygonImagery() override = default;
    OmniPolygonImagery(const OmniPolygonImagery&) = delete;
    OmniPolygonImagery& operator=(const OmniPolygonImagery&) = delete;
    OmniPolygonImagery(OmniPolygonImagery&&) noexcept = default;
    OmniPolygonImagery& operator=(OmniPolygonImagery&&) noexcept = default;

    [[nodiscard]] std::vector<PXR_NS::SdfPath> getCartographicPolygonPaths() const;

    [[nodiscard]] CesiumRasterOverlays::RasterOverlay* getRasterOverlay() const override;
    void reload() override;

  private:
    CesiumUtility::IntrusivePointer<CesiumRasterOverlays::RasterizedPolygonsOverlay> _pPolygonRasterOverlay;
};
} // namespace cesium::omniverse
