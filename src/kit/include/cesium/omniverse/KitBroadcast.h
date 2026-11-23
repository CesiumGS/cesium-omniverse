#pragma once

#include "cesium/omniverse/IBroadcast.h"

namespace cesium::omniverse {

class KitBroadcast : public IBroadcast {
  public:
    void assetsUpdated() const override;
    void connectionUpdated() const override;
    void profileUpdated() const override;
    void tokensUpdated() const override;
    void showTroubleshooter(
        const pxr::SdfPath& tilesetPath,
        int64_t tilesetIonAssetId,
        const std::string& tilesetName,
        int64_t rasterOverlayIonAssetId,
        const std::string& rasterOverlayName,
        const std::string& message) const override;
    void setDefaultTokenComplete() const override;
    void tilesetLoaded(const pxr::SdfPath& tilesetPath) const override;
    void sendMessageToBus(uint64_t eventType) const override;
    void sendMessageToBus(const std::string_view& eventKey) const override;
};

} // namespace cesium::omniverse
