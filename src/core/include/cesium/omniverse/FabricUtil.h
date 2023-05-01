#pragma once

#include "cesium/omniverse/FabricStatistics.h"

#include <glm/glm.hpp>
#include <pxr/usd/sdf/path.h>

#include <string>

namespace cesium::omniverse::FabricUtil {

std::string printFabricStage();
FabricStatistics getStatistics();
void destroyPrim(const pxr::SdfPath& path);
void destroyPrims(const std::vector<pxr::SdfPath>& paths);
void setTilesetTransform(int64_t tilesetId, const glm::dmat4& ecefToUsdTransform);
void setTilesetIdAndTileId(const pxr::SdfPath& path, int64_t tilesetId, int64_t tileId);

} // namespace cesium::omniverse::FabricUtil
