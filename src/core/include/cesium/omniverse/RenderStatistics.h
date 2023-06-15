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
    uint64_t tilesVisited{0};
    uint64_t culledTilesVisited{0};
    uint64_t tilesRendered{0};
    uint64_t tilesCulled{0};
    uint64_t maxDepthVisited{0};
    uint64_t tilesLoadingWorker{0};
    uint64_t tilesLoadingMain{0};
    uint64_t tilesLoaded{0};
};

} // namespace cesium::omniverse
