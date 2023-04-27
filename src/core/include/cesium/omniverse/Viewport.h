#pragma once

#include <pxr/base/gf/matrix4d.h>

namespace cesium::omniverse {

struct Viewport {
    pxr::GfMatrix4d viewMatrix;
    pxr::GfMatrix4d projMatrix;
    double width;
    double height;
};

} // namespace cesium::omniverse
