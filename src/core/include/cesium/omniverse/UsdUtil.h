#pragma once

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
glm::dmat4 usdToGlmMatrix(const pxr::GfMatrix4d& matrix);
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

}; // namespace cesium::omniverse::UsdUtil
