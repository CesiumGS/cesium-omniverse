#pragma once

#include <CesiumIonClient/Token.h>
#include <pxr/usd/sdf/path.h>

namespace cesium::omniverse {
class OmniImagery {
  public:
    OmniImagery(pxr::SdfPath path);

    [[nodiscard]] pxr::SdfPath getPath() const;
    [[nodiscard]] std::string getName() const;
    [[nodiscard]] int64_t getIonAssetId() const;
    [[nodiscard]] std::optional<CesiumIonClient::Token> getIonAccessToken() const;
    [[nodiscard]] bool getShowCreditsOnScreen() const;

  private:
    pxr::SdfPath _path;
};
} // namespace cesium::omniverse
