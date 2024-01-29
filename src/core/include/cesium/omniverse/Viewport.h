#pragma once

#include <glm/glm.hpp>

namespace cesium::omniverse {

struct Viewport {
    glm::dmat4 viewMatrix;
    glm::dmat4 projMatrix;
    double width;
    double height;
};

} // namespace cesium::omniverse
