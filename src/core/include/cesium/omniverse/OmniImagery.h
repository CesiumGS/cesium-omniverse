#pragma once

#include "cesium/omniverse/GltfUtil.h"

#include <CesiumIonClient/Token.h>
#include <pxr/usd/sdf/path.h>

namespace cesium::omniverse {
class OmniImagery {
  public:
    OmniImagery(const pxr::SdfPath& path);
    [[nodiscard]] pxr::SdfPath getPath() const;
    [[nodiscard]] std::string getName() const;
    [[nodiscard]] bool getShowCreditsOnScreen() const;
    [[nodiscard]] double getAlpha() const;
    [[nodiscard]] OverlayRenderMethod getOverlayRenderMethod() const;

  protected:
    pxr::SdfPath _path;
};
} // namespace cesium::omniverse
