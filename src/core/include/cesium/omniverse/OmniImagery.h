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
    [[nodiscard]] bool getShowCreditsOnScreen() const;
    [[nodiscard]] double getAlpha() const;

  protected:
    pxr::SdfPath _path;
};
} // namespace cesium::omniverse
