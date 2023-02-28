#pragma once

#include <CesiumIonClient/Token.h>
#include <pxr/usd/sdf/path.h>

namespace cesium::omniverse {
class OmniIonRasterOverlay {
  public:
    OmniIonRasterOverlay(const pxr::SdfPath& path);

    pxr::SdfPath getPath() const;
    std::string getName() const;
    int64_t getIonAssetId() const;
    std::optional<CesiumIonClient::Token> getIonAccessToken() const;

  private:
    pxr::SdfPath _path;
};
} // namespace cesium::omniverse
