#pragma once

#include <CesiumUsdSchemas/data.h>
#include <CesiumUsdSchemas/rasterOverlay.h>
#include <CesiumUsdSchemas/tileset.h>
#include <carb/flatcache/StageWithHistory.h>
#include <glm/glm.hpp>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/quatf.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/common.h>

namespace CesiumGeospatial {
class Cartographic;
}

namespace cesium::omniverse::UsdUtil {

struct Decomposed {
    pxr::GfVec3d position;
    pxr::GfQuatf orientation;
    pxr::GfVec3f scale;
};

pxr::UsdStageRefPtr getUsdStage();
carb::flatcache::StageInProgress getFabricStageInProgress();
bool hasStage();
glm::dvec3 usdToGlmVector(const pxr::GfVec3d& vector);
glm::dmat4 usdToGlmMatrix(const pxr::GfMatrix4d& matrix);
pxr::GfVec3d glmToUsdVector(const glm::dvec3& vector);
pxr::GfMatrix4d glmToUsdMatrix(const glm::dmat4& matrix);
Decomposed glmToUsdMatrixDecomposed(const glm::dmat4& matrix);
glm::dmat4 computeUsdWorldTransform(const pxr::SdfPath& path);
bool isPrimVisible(const pxr::SdfPath& path);
pxr::TfToken getUsdUpAxis();
double getUsdMetersPerUnit();
pxr::SdfPath getRootPath();
pxr::SdfPath getPathUnique(const pxr::SdfPath& parentPath, const std::string& name);
std::string getSafeName(const std::string& name);
glm::dmat4 computeUsdToEcefTransform(const CesiumGeospatial::Cartographic& origin);
glm::dmat4 computeEcefToUsdTransform(const CesiumGeospatial::Cartographic& origin);
glm::dmat4 computeEcefToUsdTransformForPrim(const CesiumGeospatial::Cartographic& origin, const pxr::SdfPath& primPath);
pxr::GfRange3d computeWorldExtent(const pxr::GfRange3d& localExtent, const glm::dmat4& localToUsdTransform);

pxr::CesiumData defineCesiumData(const pxr::SdfPath& path);
pxr::CesiumTileset defineCesiumTileset(const pxr::SdfPath& path);
pxr::CesiumRasterOverlay defineCesiumRasterOverlay(const pxr::SdfPath& path);

pxr::CesiumData getCesiumData(const pxr::SdfPath& path);
pxr::CesiumTileset getCesiumTileset(const pxr::SdfPath& path);
pxr::CesiumRasterOverlay getCesiumRasterOverlay(const pxr::SdfPath& path);
std::vector<pxr::SdfPath> getChildRasterOverlayPaths(const pxr::SdfPath& path);

bool isCesiumData(const pxr::SdfPath& path);
bool isCesiumTileset(const pxr::SdfPath& path);
bool isCesiumRasterOverlay(const pxr::SdfPath& path);

bool primExists(const pxr::SdfPath& path);

}; // namespace cesium::omniverse::UsdUtil
