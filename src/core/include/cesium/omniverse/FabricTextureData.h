#pragma once

#include <carb/RenderingTypes.h>

#include <vector>

namespace cesium::omniverse {

struct FabricTextureData {
    std::vector<std::byte> bytes;
    uint64_t width;
    uint64_t height;
    carb::Format format;
};

} // namespace cesium::omniverse
