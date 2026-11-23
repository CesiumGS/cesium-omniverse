#pragma once

#include <cstdint>
#include <string>

#ifdef CESIUM_OMNI_MSVC
#pragma warning(push)
#pragma warning(disable : 4244 4305 4100 4201 4127 4251)
#endif

#include <pxr/usd/usd/common.h>

#ifdef CESIUM_OMNI_MSVC
#pragma warning(pop)
#endif

namespace cesium::omniverse {

class IBroadcast {
  public:
    virtual void assetsUpdated() const = 0;
    virtual void connectionUpdated() const = 0;
    virtual void profileUpdated() const = 0;
    virtual void tokensUpdated() const = 0;
    virtual void showTroubleshooter(
        const pxr::SdfPath& tilesetPath,
        int64_t tilesetIonAssetId,
        const std::string& tilesetName,
        int64_t rasterOverlayIonAssetId,
        const std::string& rasterOverlayName,
        const std::string& message) const = 0;
    virtual void setDefaultTokenComplete() const = 0;
    virtual void tilesetLoaded(const pxr::SdfPath& tilesetPath) const = 0;
    virtual void sendMessageToBus(uint64_t eventType) const = 0;
    virtual void sendMessageToBus(const std::string_view& eventKey) const = 0;
};

} // namespace cesium::omniverse
