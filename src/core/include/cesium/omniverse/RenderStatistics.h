#pragma once

#include <cstdint>

namespace cesium::omniverse {

struct RenderStatistics {
    uint64_t numberOfMaterialsLoaded{0};
    uint64_t numberOfGeometriesLoaded{0};
    uint64_t numberOfGeometriesVisible{0};
    uint64_t numberOfTrianglesLoaded{0};
    uint64_t numberOfTrianglesVisible{0};
    uint64_t tilesetCachedBytes{0};
};

} // namespace cesium::omniverse
