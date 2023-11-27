#pragma once

#include "cesium/omniverse/OmniImagery.h"
#include <CesiumIonClient/Token.h>
#include <pxr/usd/sdf/path.h>

namespace cesium::omniverse {
class OmniIonImagery : public OmniImagery {
public:
    OmniIonImagery(const pxr::SdfPath& path) : OmniImagery(path) {}
};
} // namespace cesium::omniverse
