#pragma once

#include <glm/glm.hpp>

#include <vector>

namespace cesium::omniverse {

struct FabricTextureInfo {
    glm::dvec2 offset;
    double rotation;
    glm::dvec2 scale;
    uint64_t setIndex;
    int32_t wrapS;
    int32_t wrapT;
    bool flipVertical;
    std::vector<uint8_t> channels;

    bool operator==(const FabricTextureInfo& other) const = default;
};

} // namespace cesium::omniverse
