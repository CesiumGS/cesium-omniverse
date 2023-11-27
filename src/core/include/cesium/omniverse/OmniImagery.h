#pragma once

#include <CesiumIonClient/Token.h>
#include <pxr/usd/sdf/path.h>

namespace cesium::omniverse {
class OmniImagery {
  protected:
    OmniImagery(const pxr::SdfPath& path);

  public:
    [[nodiscard]] pxr::SdfPath getPath() const;
    [[nodiscard]] std::string getName() const;
    [[nodiscard]] int64_t getIonAssetId() const;
    [[nodiscard]] std::optional<CesiumIonClient::Token> getIonAccessToken() const;
    [[nodiscard]] bool getShowCreditsOnScreen() const;
    [[nodiscard]] double getAlpha() const;

  private:
    pxr::SdfPath _path;
};
} // namespace cesium::omniverse
