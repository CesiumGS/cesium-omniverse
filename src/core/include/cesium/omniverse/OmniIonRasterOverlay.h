#pragma once

#include "cesium/omniverse/OmniRasterOverlay.h"

#include <CesiumRasterOverlays/IonRasterOverlay.h>
#include <CesiumUtility/IntrusivePointer.h>

namespace CesiumIonClient {
struct Token;
}

namespace cesium::omniverse {

class OmniIonRasterOverlay final : public OmniRasterOverlay {
  public:
    OmniIonRasterOverlay(Context* pContext, const pxr::SdfPath& path);
    ~OmniIonRasterOverlay() override = default;
    OmniIonRasterOverlay(const OmniIonRasterOverlay&) = delete;
    OmniIonRasterOverlay& operator=(const OmniIonRasterOverlay&) = delete;
    OmniIonRasterOverlay(OmniIonRasterOverlay&&) noexcept = default;
    OmniIonRasterOverlay& operator=(OmniIonRasterOverlay&&) noexcept = default;

    [[nodiscard]] int64_t getIonAssetId() const;
    [[nodiscard]] CesiumIonClient::Token getIonAccessToken() const;
    [[nodiscard]] std::string getIonApiUrl() const;
    [[nodiscard]] pxr::SdfPath getResolvedIonServerPath() const;

    [[nodiscard]] CesiumRasterOverlays::RasterOverlay* getRasterOverlay() const override;
    void reload() override;

  private:
    CesiumUtility::IntrusivePointer<CesiumRasterOverlays::IonRasterOverlay> _pIonRasterOverlay;
};
} // namespace cesium::omniverse
