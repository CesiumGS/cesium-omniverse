#pragma once

#include "DataType.h"

#include <cstdint>
#include <vector>

namespace cesium::omniverse {

struct TextureData {
    std::vector<std::byte> bytes;
    uint64_t width;
    uint64_t height;
    DataType format;
};

} // namespace cesium::omniverse
