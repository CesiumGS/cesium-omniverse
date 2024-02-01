#pragma once

#include "cesium/omniverse/OmniRasterOverlay.h"

#include <Cesium3DTilesSelection/RasterizedPolygonsTileExcluder.h>
#include <CesiumRasterOverlays/RasterizedPolygonsOverlay.h>
#include <CesiumUtility/IntrusivePointer.h>

namespace cesium::omniverse {

class OmniPolygonRasterOverlay final : public OmniRasterOverlay {
  public:
    OmniPolygonRasterOverlay(Context* pContext, const pxr::SdfPath& path);
    ~OmniPolygonRasterOverlay() override = default;
    OmniPolygonRasterOverlay(const OmniPolygonRasterOverlay&) = delete;
    OmniPolygonRasterOverlay& operator=(const OmniPolygonRasterOverlay&) = delete;
    OmniPolygonRasterOverlay(OmniPolygonRasterOverlay&&) noexcept = default;
    OmniPolygonRasterOverlay& operator=(OmniPolygonRasterOverlay&&) noexcept = default;

    [[nodiscard]] std::vector<pxr::SdfPath> getCartographicPolygonPaths() const;
    [[nodiscard]] CesiumRasterOverlays::RasterOverlay* getRasterOverlay() const override;
    [[nodiscard]] bool getInvertSelection() const;
    [[nodiscard]] bool getExcludeSelectedTiles() const;
    void reload() override;

  private:
    CesiumUtility::IntrusivePointer<CesiumRasterOverlays::RasterizedPolygonsOverlay> _pPolygonRasterOverlay;
    std::shared_ptr<Cesium3DTilesSelection::RasterizedPolygonsTileExcluder> _pExcluder;
};
} // namespace cesium::omniverse
