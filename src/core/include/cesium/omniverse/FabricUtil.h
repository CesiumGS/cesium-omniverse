#pragma once

#include "cesium/omniverse/RenderStatistics.h"

#include <glm/glm.hpp>
#include <pxr/usd/sdf/path.h>

#include <string>

namespace cesium::omniverse {

struct FabricStatistics {
    uint64_t numberOfMaterialsLoaded{0};
    uint64_t numberOfGeometriesLoaded{0};
    uint64_t numberOfGeometriesVisible{0};
    uint64_t numberOfTrianglesLoaded{0};
    uint64_t numberOfTrianglesVisible{0};
};

} // namespace cesium::omniverse

namespace cesium::omniverse::FabricUtil {

std::string printFabricStage();
FabricStatistics getStatistics();
void destroyPrim(const pxr::SdfPath& path);
void destroyPrims(const std::vector<pxr::SdfPath>& paths);
void setTilesetTransform(int64_t tilesetId, const glm::dmat4& ecefToUsdTransform);
void setTilesetIdAndTileId(const pxr::SdfPath& path, int64_t tilesetId, int64_t tileId);

} // namespace cesium::omniverse::FabricUtil
