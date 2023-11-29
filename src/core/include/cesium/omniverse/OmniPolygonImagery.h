#pragma once

#include "cesium/omniverse/OmniImagery.h"

#include <CesiumIonClient/Token.h>
#include <pxr/usd/sdf/path.h>

namespace cesium::omniverse {
class OmniPolygonImagery : public OmniImagery {
  public:
    OmniPolygonImagery(const pxr::SdfPath& path);
};
} // namespace cesium::omniverse
