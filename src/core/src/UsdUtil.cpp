#include "cesium/omniverse/UsdUtil.h"

#include "cesium/omniverse/Context.h"

#include <CesiumGeometry/AxisTransforms.h>
#include <CesiumGeospatial/Cartographic.h>
#include <CesiumGeospatial/Transforms.h>
#include <glm/gtx/matrix_decompose.hpp>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/timeCode.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdUtils/stageCache.h>
#include <spdlog/fmt/fmt.h>

#include <regex>

namespace cesium::omniverse::UsdUtil {

namespace {
glm::dmat4 getEastNorthUpToFixedFrame(const CesiumGeospatial::Cartographic& cartographic) {
    const auto cartesian = CesiumGeospatial::Ellipsoid::WGS84.cartographicToCartesian(cartographic);
    const auto matrix = CesiumGeospatial::Transforms::eastNorthUpToFixedFrame(cartesian);
    return matrix;
}

glm::dmat4 getAxisConversionTransform() {
    const auto upAxis = getUsdUpAxis();

    auto axisConversion = glm::dmat4(1.0);

    // USD up axis can be either Y or Z
    if (upAxis == pxr::UsdGeomTokens->y) {
        axisConversion = CesiumGeometry::AxisTransforms::Y_UP_TO_Z_UP;
    }

    return axisConversion;
}

glm::dmat4 getUnitConversionTransform() {
    const auto metersPerUnit = getUsdMetersPerUnit();
    const auto matrix = glm::scale(glm::dmat4(1.0), glm::dvec3(metersPerUnit));
    return matrix;
}

} // namespace

pxr::UsdStageRefPtr getUsdStage() {
    const auto stageId = Context::instance().getStageId();
    return pxr::UsdUtilsStageCache::Get().Find(pxr::UsdStageCache::Id::FromLongInt(stageId));
}

carb::flatcache::StageInProgress getFabricStageInProgress() {
    const auto stageId = Context::instance().getStageId();
    const auto iStageInProgress = carb::getCachedInterface<carb::flatcache::IStageInProgress>();
    const auto stageInProgressId = iStageInProgress->get(carb::flatcache::UsdStageId{static_cast<uint64_t>(stageId)});
    return carb::flatcache::StageInProgress(stageInProgressId);
}

glm::dmat4 usdToGlmMatrix(const pxr::GfMatrix4d& matrix) {
    // Row-major to column-major
    return glm::dmat4{
        matrix[0][0],
        matrix[1][0],
        matrix[2][0],
        matrix[3][0],
        matrix[0][1],
        matrix[1][1],
        matrix[2][1],
        matrix[3][1],
        matrix[0][2],
        matrix[1][2],
        matrix[2][2],
        matrix[3][2],
        matrix[0][3],
        matrix[1][3],
        matrix[2][3],
        matrix[3][3],
    };
}

pxr::GfMatrix4d glmToUsdMatrix(const glm::dmat4& matrix) {
    // Column-major to row-major
    return pxr::GfMatrix4d{
        matrix[0][0],
        matrix[1][0],
        matrix[2][0],
        matrix[3][0],
        matrix[0][1],
        matrix[1][1],
        matrix[2][1],
        matrix[3][1],
        matrix[0][2],
        matrix[1][2],
        matrix[2][2],
        matrix[3][2],
        matrix[0][3],
        matrix[1][3],
        matrix[2][3],
        matrix[3][3],
    };
}

Decomposed glmToUsdMatrixDecomposed(const glm::dmat4& matrix) {
    glm::dvec3 scale{};
    glm::dquat rotation{};
    glm::dvec3 translation{};
    glm::dvec3 skew{};
    glm::dvec4 perspective{};

    [[maybe_unused]] const auto decomposable = glm::decompose(matrix, scale, rotation, translation, skew, perspective);
    assert(decomposable);

    const glm::fquat rotationF32(rotation);
    const glm::fvec3 scaleF32(scale);

    return {
        pxr::GfVec3d(translation.x, translation.y, translation.z),
        pxr::GfQuatf(rotationF32.w, pxr::GfVec3f(rotationF32.x, rotationF32.y, rotationF32.z)),
        pxr::GfVec3f(scaleF32.x, scaleF32.y, scaleF32.z),
    };
}

glm::dmat4 computeUsdWorldTransform(const pxr::SdfPath& path) {
    const auto stage = getUsdStage();
    const auto prim = stage->GetPrimAtPath(path);
    assert(prim.IsValid());
    const auto xform = pxr::UsdGeomXformable(prim);
    const auto time = pxr::UsdTimeCode::Default();
    const auto transform = xform.ComputeLocalToWorldTransform(time);
    const auto matrix = usdToGlmMatrix(transform);

    // For some reason the USD matrix is column major instead of row major, so we need to transpose here
    return glm::transpose(matrix);
}

bool isPrimVisible(const pxr::SdfPath& path) {
    // This is similar to isPrimVisible in kit-sdk/dev/include/omni/usd/UsdUtils.h
    const auto stage = getUsdStage();
    const auto prim = stage->GetPrimAtPath(path);
    assert(prim.IsValid());
    const auto imageable = pxr::UsdGeomImageable(prim);
    const auto time = pxr::UsdTimeCode::Default();
    const auto visibility = imageable.ComputeVisibility(time);
    return visibility != pxr::UsdGeomTokens->invisible;
}

pxr::TfToken getUsdUpAxis() {
    const auto stage = getUsdStage();
    const auto upAxis = pxr::UsdGeomGetStageUpAxis(stage);
    return upAxis;
}

double getUsdMetersPerUnit() {
    const auto stage = getUsdStage();
    const auto metersPerUnit = pxr::UsdGeomGetStageMetersPerUnit(stage);
    return metersPerUnit;
}

pxr::SdfPath getRootPath() {
    const auto stage = getUsdStage();
    return stage->GetPseudoRoot().GetPath();
}

pxr::SdfPath getPathUnique(const pxr::SdfPath& parentPath, const std::string& name) {
    const auto stage = getUsdStage();
    pxr::UsdPrim prim;
    pxr::SdfPath path;
    auto copy = 0;

    do {
        const auto copyName = copy > 0 ? fmt::format("{}_{}", name, copy) : name;
        path = parentPath.AppendChild(pxr::TfToken(copyName));
        prim = stage->GetPrimAtPath(path);
        copy++;
    } while (prim.IsValid());

    return path;
}

std::string getSafeName(const std::string& assetName) {
    const std::regex regex("[\\W]+");
    const std::string replace = "_";

    return std::regex_replace(assetName, regex, replace);
}

glm::dmat4 computeUsdToEcefTransform(const CesiumGeospatial::Cartographic& origin) {
    return getEastNorthUpToFixedFrame(origin) * getAxisConversionTransform() * getUnitConversionTransform();
}

glm::dmat4 computeEcefToUsdTransform(const CesiumGeospatial::Cartographic& origin) {
    return glm::inverse(computeUsdToEcefTransform(origin));
}

glm::dmat4
computeEcefToUsdTransformForPrim(const CesiumGeospatial::Cartographic& origin, const pxr::SdfPath& primPath) {
    const auto ecefToUsdTransform = computeEcefToUsdTransform(origin);
    const auto primUsdWorldTransform = computeUsdWorldTransform(primPath);
    const auto primEcefToUsdTransform = primUsdWorldTransform * ecefToUsdTransform;
    return primEcefToUsdTransform;
}

} // namespace cesium::omniverse::UsdUtil
