#pragma once

#include <vector>

namespace cesium::omniverse {

enum class FabricOverlayRenderMethod {
    OVERLAY = 0,
    CLIPPING = 1,
};

struct FabricRasterOverlaysInfo {
    std::vector<FabricOverlayRenderMethod> overlayRenderMethods;
};

} // namespace cesium::omniverse
