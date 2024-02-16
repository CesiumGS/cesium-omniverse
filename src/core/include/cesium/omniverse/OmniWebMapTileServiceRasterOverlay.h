#pragma once

#include "cesium/omniverse/OmniRasterOverlay.h"

#include <CesiumRasterOverlays/WebMapTileServiceRasterOverlay.h>
#include <CesiumUtility/IntrusivePointer.h>

#include <string>

namespace cesium::omniverse {

class OmniWebMapTileServiceRasterOverlay final : public OmniRasterOverlay {
  public:
    OmniWebMapTileServiceRasterOverlay(Context* pContext, const pxr::SdfPath& path);
    ~OmniWebMapTileServiceRasterOverlay() override = default;
    OmniWebMapTileServiceRasterOverlay(const OmniWebMapTileServiceRasterOverlay&) = delete;
    OmniWebMapTileServiceRasterOverlay& operator=(const OmniWebMapTileServiceRasterOverlay&) = delete;
    OmniWebMapTileServiceRasterOverlay(OmniWebMapTileServiceRasterOverlay&&) noexcept = default;
    OmniWebMapTileServiceRasterOverlay& operator=(OmniWebMapTileServiceRasterOverlay&&) noexcept = default;

    [[nodiscard]] CesiumRasterOverlays::RasterOverlay* getRasterOverlay() const override;
    [[nodiscard]] std::string getUrl() const;
    [[nodiscard]] std::string getLayer() const;
    [[nodiscard]] std::string getTileMatrixSetId() const;
    [[nodiscard]] std::string getStyle() const;
    [[nodiscard]] std::string getFormat() const;
    [[nodiscard]] int getMinimumZoomLevel() const;
    [[nodiscard]] int getMaximumZoomLevel() const;
    [[nodiscard]] bool getSpecifyZoomLevels() const;
    [[nodiscard]] bool getUseWebMercatorProjection() const;
    [[nodiscard]] bool getSpecifyTilingScheme() const;
    [[nodiscard]] double getNorth() const;
    [[nodiscard]] double getSouth() const;
    [[nodiscard]] double getEast() const;
    [[nodiscard]] double getWest() const;
    [[nodiscard]] bool getSpecifyTileMatrixSetLabels() const;
    [[nodiscard]] std::string getTileMatrixSetLabelPrefix() const;
    [[nodiscard]] std::vector<std::string> getTileMatrixSetLabels() const;
    [[nodiscard]] int getRootTilesX() const;
    [[nodiscard]] int getRootTilesY() const;

    void reload() override;

  private:
    CesiumUtility::IntrusivePointer<CesiumRasterOverlays::WebMapTileServiceRasterOverlay>
        _pWebMapTileServiceRasterOverlay;
};
} // namespace cesium::omniverse
