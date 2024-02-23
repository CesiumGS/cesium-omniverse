#pragma once

#include "cesium/omniverse/OmniRasterOverlay.h"

#include <CesiumRasterOverlays/TileMapServiceRasterOverlay.h>
#include <CesiumUtility/IntrusivePointer.h>

#include <string>

namespace cesium::omniverse {

class OmniTileMapServiceRasterOverlay final : public OmniRasterOverlay {
  public:
    OmniTileMapServiceRasterOverlay(Context* pContext, const pxr::SdfPath& path);
    ~OmniTileMapServiceRasterOverlay() override = default;
    OmniTileMapServiceRasterOverlay(const OmniTileMapServiceRasterOverlay&) = delete;
    OmniTileMapServiceRasterOverlay& operator=(const OmniTileMapServiceRasterOverlay&) = delete;
    OmniTileMapServiceRasterOverlay(OmniTileMapServiceRasterOverlay&&) noexcept = default;
    OmniTileMapServiceRasterOverlay& operator=(OmniTileMapServiceRasterOverlay&&) noexcept = default;

    [[nodiscard]] CesiumRasterOverlays::RasterOverlay* getRasterOverlay() const override;
    [[nodiscard]] std::string getUrl() const;
    [[nodiscard]] int getMinimumZoomLevel() const;
    [[nodiscard]] int getMaximumZoomLevel() const;
    [[nodiscard]] bool getSpecifyZoomLevels() const;
    void reload() override;

  private:
    CesiumUtility::IntrusivePointer<CesiumRasterOverlays::TileMapServiceRasterOverlay> _pTileMapServiceRasterOverlay;
};
} // namespace cesium::omniverse
