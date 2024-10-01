#pragma once

#include "cesium/omniverse/OmniTileset.h"

#include <CesiumRasterOverlays/RasterOverlay.h>
#include <pxr/usd/sdf/path.h>

namespace CesiumRasterOverlays {
class RasterOverlay;
}

namespace cesium::omniverse {

class Context;
class OmniTileset;
enum class FabricOverlayRenderMethod;

class OmniRasterOverlay {
    friend void OmniTileset::addRasterOverlayIfExists(const OmniRasterOverlay* pOverlay);
    friend pxr::SdfPath
    OmniTileset::getRasterOverlayPathIfExists(const CesiumRasterOverlays::RasterOverlay& rasterOverlay);

  public:
    OmniRasterOverlay(Context* pContext, const pxr::SdfPath& path);
    virtual ~OmniRasterOverlay() = default;
    OmniRasterOverlay(const OmniRasterOverlay&) = delete;
    OmniRasterOverlay& operator=(const OmniRasterOverlay&) = delete;
    OmniRasterOverlay(OmniRasterOverlay&&) noexcept = default;
    OmniRasterOverlay& operator=(OmniRasterOverlay&&) noexcept = default;

    [[nodiscard]] const pxr::SdfPath& getPath() const;
    [[nodiscard]] bool getShowCreditsOnScreen() const;
    [[nodiscard]] double getAlpha() const;
    [[nodiscard]] FabricOverlayRenderMethod getOverlayRenderMethod() const;
    [[nodiscard]] float getMaximumScreenSpaceError() const;
    [[nodiscard]] int getMaximumTextureSize() const;
    [[nodiscard]] int getMaximumSimultaneousTileLoads() const;
    [[nodiscard]] int getSubTileCacheBytes() const;

    [[nodiscard]] CesiumRasterOverlays::RasterOverlayOptions createRasterOverlayOptions() const;

    void updateRasterOverlayOptions() const;
    virtual void reload() = 0;

  protected:
    [[nodiscard]] virtual CesiumRasterOverlays::RasterOverlay* getRasterOverlay() const = 0;
    [[nodiscard]] const CesiumGeospatial::Ellipsoid& getEllipsoid() const;

    Context* _pContext;
    pxr::SdfPath _path;

  private:
    void setRasterOverlayOptionsFromUsd(CesiumRasterOverlays::RasterOverlayOptions& options) const;
};
} // namespace cesium::omniverse
