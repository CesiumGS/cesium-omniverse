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

    // Make sure to update this function when adding new fields to the struct
    // In C++ 20 we can use the default equality comparison (= default)
    // clang-format off
    bool operator==(const FabricTextureInfo& other) const {
        return offset == other.offset &&
               rotation == other.rotation &&
               scale == other.scale &&
               setIndex == other.setIndex &&
               wrapS == other.wrapS &&
               wrapT == other.wrapT &&
               flipVertical == other.flipVertical &&
               channels == other.channels;
    }
    // clang-format on
};

} // namespace cesium::omniverse
