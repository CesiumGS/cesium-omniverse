#pragma once

#include "cesium/omniverse/OmniImagery.h"

#include <CesiumRasterOverlays/IonRasterOverlay.h>
#include <CesiumUtility/IntrusivePointer.h>

namespace CesiumIonClient {
struct Token;
}

namespace cesium::omniverse {

class OmniIonImagery final : public OmniImagery {
  public:
    OmniIonImagery(Context* pContext, const pxr::SdfPath& path);
    ~OmniIonImagery() override = default;
    OmniIonImagery(const OmniIonImagery&) = delete;
    OmniIonImagery& operator=(const OmniIonImagery&) = delete;
    OmniIonImagery(OmniIonImagery&&) noexcept = default;
    OmniIonImagery& operator=(OmniIonImagery&&) noexcept = default;

    [[nodiscard]] int64_t getIonAssetId() const;
    [[nodiscard]] CesiumIonClient::Token getIonAccessToken() const;
    [[nodiscard]] std::string getIonApiUrl() const;
    [[nodiscard]] pxr::SdfPath getIonServerPath() const;

    [[nodiscard]] CesiumRasterOverlays::RasterOverlay* getRasterOverlay() const override;
    void reload() override;

    void setIonServerPath(const pxr::SdfPath& ionServerPath);

  private:
    CesiumUtility::IntrusivePointer<CesiumRasterOverlays::IonRasterOverlay> _pIonRasterOverlay;
};
} // namespace cesium::omniverse
