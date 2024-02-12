#pragma once

#include "cesium/omniverse/OmniRasterOverlay.h"

#include <CesiumRasterOverlays/WebMapServiceRasterOverlay.h>
#include <CesiumUtility/IntrusivePointer.h>

#include <string>

namespace cesium::omniverse {

class OmniWebMapServiceRasterOverlay final : public OmniRasterOverlay {
  public:
    OmniWebMapServiceRasterOverlay(Context* pContext, const pxr::SdfPath& path);
    ~OmniWebMapServiceRasterOverlay() override = default;
    OmniWebMapServiceRasterOverlay(const OmniWebMapServiceRasterOverlay&) = delete;
    OmniWebMapServiceRasterOverlay& operator=(const OmniWebMapServiceRasterOverlay&) = delete;
    OmniWebMapServiceRasterOverlay(OmniWebMapServiceRasterOverlay&&) noexcept = default;
    OmniWebMapServiceRasterOverlay& operator=(OmniWebMapServiceRasterOverlay&&) noexcept = default;

    [[nodiscard]] CesiumRasterOverlays::RasterOverlay* getRasterOverlay() const override;
    [[nodiscard]] std::string getBaseUrl() const;
    [[nodiscard]] int getMinimumLevel() const;
    [[nodiscard]] int getMaximumLevel() const;
    [[nodiscard]] int getTileWidth() const;
    [[nodiscard]] int getTileHeight() const;
    [[nodiscard]] std::string getLayers() const;
    void reload() override;

  private:
    CesiumUtility::IntrusivePointer<CesiumRasterOverlays::WebMapServiceRasterOverlay> _pWebMapServiceRasterOverlay;
};
} // namespace cesium::omniverse
