#pragma once

#include "cesium/omniverse/RenderStatistics.h"

#include <carb/flatcache/IPath.h>
#include <glm/glm.hpp>
#include <pxr/usd/sdf/path.h>

#include <string>

namespace cesium::omniverse {

struct FabricStatistics {
    uint64_t materialsCapacity{0};
    uint64_t materialsLoaded{0};
    uint64_t geometriesCapacity{0};
    uint64_t geometriesLoaded{0};
    uint64_t geometriesRendered{0};
    uint64_t trianglesLoaded{0};
    uint64_t trianglesRendered{0};
};

} // namespace cesium::omniverse

namespace cesium::omniverse::FabricUtil {

std::string printFabricStage();
FabricStatistics getStatistics();
void destroyPrim(const carb::flatcache::Path& path);
void destroyPrims(const std::vector<carb::flatcache::Path>& paths);
void setTilesetTransform(int64_t tilesetId, const glm::dmat4& ecefToUsdTransform);
void setTilesetIdAndTileId(const carb::flatcache::Path& pathFabric, int64_t tilesetId, int64_t tileId);

} // namespace cesium::omniverse::FabricUtil
