#pragma once

#include <cstdint>

namespace cesium::omniverse {

struct RenderStatistics {
    uint64_t materialsCapacity{0};
    uint64_t materialsLoaded{0};
    uint64_t geometriesCapacity{0};
    uint64_t geometriesLoaded{0};
    uint64_t geometriesRendered{0};
    uint64_t trianglesLoaded{0};
    uint64_t trianglesRendered{0};
    uint64_t tilesetCachedBytes{0};
};

} // namespace cesium::omniverse
