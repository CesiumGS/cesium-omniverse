#pragma once

#include <vector>

namespace cesium::omniverse {

enum class OverlayRenderMethod {
    OVERLAY = 0,
    CLIPPING = 1,
};

struct RasterOverlaysInfo {
    std::vector<OverlayRenderMethod> overlayRenderMethods;
};

} // namespace cesium::omniverse
