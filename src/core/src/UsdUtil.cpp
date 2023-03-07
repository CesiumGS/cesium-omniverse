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
    return Context::instance().getStage();
}

carb::flatcache::StageInProgress getFabricStageInProgress() {
    return Context::instance().getFabricStageInProgress();
}

bool hasStage() {
    return Context::instance().getStageId() != 0;
}

glm::dvec3 usdToGlmVector(const pxr::GfVec3d& vector) {
    return glm::dvec3(vector[0], vector[1], vector[2]);
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

pxr::GfVec3d glmToUsdVector(const glm::dvec3& vector) {
    return pxr::GfVec3d(vector.x, vector.y, vector.z);
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

pxr::GfRange3d computeWorldExtent(const pxr::GfRange3d& localExtent, const glm::dmat4& localToUsdTransform) {
    const auto min = std::numeric_limits<double>::lowest();
    const auto max = std::numeric_limits<double>::max();

    glm::dvec3 worldMin(max);
    glm::dvec3 worldMax(min);

    for (int i = 0; i < 8; i++) {
        const auto localPosition = usdToGlmVector(localExtent.GetCorner(i));
        const auto worldPosition = glm::dvec3(localToUsdTransform * glm::dvec4(localPosition, 1.0));

        worldMin = glm::min(worldMin, worldPosition);
        worldMax = glm::max(worldMax, worldPosition);
    }

    return pxr::GfRange3d(glmToUsdVector(worldMin), glmToUsdVector(worldMax));
}

pxr::CesiumData defineCesiumData(const pxr::SdfPath& path) {
    auto stage = getUsdStage();
    auto cesiumData = pxr::CesiumData::Define(stage, path);

    cesiumData.CreateDefaultProjectIonAccessTokenAttr();
    cesiumData.CreateDefaultProjectIonAccessTokenIdAttr();
    cesiumData.CreateGeoreferenceOriginLongitudeAttr(pxr::VtValue(-105.25737));
    cesiumData.CreateGeoreferenceOriginLatitudeAttr(pxr::VtValue(39.736401));
    cesiumData.CreateGeoreferenceOriginHeightAttr(pxr::VtValue(2250.0));

    return cesiumData;
}

pxr::CesiumTilesetAPI defineCesiumTileset(const pxr::SdfPath& path) {
    auto stage = getUsdStage();
    auto xform = pxr::UsdGeomXform::Define(stage, path);
    assert(xform.GetPrim().IsValid());

    auto tileset = pxr::CesiumTilesetAPI::Apply(xform.GetPrim());
    assert(tileset.GetPrim().IsValid());

    tileset.CreateUrlAttr();
    tileset.CreateIonAssetIdAttr();
    tileset.CreateIonAccessTokenAttr();
    tileset.CreateMaximumScreenSpaceErrorAttr(pxr::VtValue(16.0f));
    tileset.CreatePreloadAncestorsAttr(pxr::VtValue(true));
    tileset.CreatePreloadSiblingsAttr(pxr::VtValue(true));
    tileset.CreateForbidHolesAttr(pxr::VtValue(false));
    tileset.CreateMaximumSimultaneousTileLoadsAttr(pxr::VtValue(uint32_t(20)));
    tileset.CreateMaximumCachedBytesAttr(pxr::VtValue(uint64_t(536870912)));
    tileset.CreateLoadingDescendantLimitAttr(pxr::VtValue(uint32_t(20)));
    tileset.CreateEnableFrustumCullingAttr(pxr::VtValue(true));
    tileset.CreateEnableFogCullingAttr(pxr::VtValue(true));
    tileset.CreateEnforceCulledScreenSpaceErrorAttr(pxr::VtValue(true));
    tileset.CreateCulledScreenSpaceErrorAttr(pxr::VtValue(64.0f));
    tileset.CreateSuspendUpdateAttr(pxr::VtValue(false));

    return tileset;
}

pxr::CesiumRasterOverlay defineCesiumRasterOverlay(const pxr::SdfPath& path) {
    auto stage = getUsdStage();
    auto rasterOverlay = pxr::CesiumRasterOverlay::Define(stage, path);
    assert(rasterOverlay.GetPrim().IsValid());

    rasterOverlay.CreateIonAssetIdAttr();
    rasterOverlay.CreateIonAccessTokenAttr();

    return rasterOverlay;
}

pxr::CesiumData getOrCreateCesiumData() {
    static const auto CesiumDataPath = pxr::SdfPath("/Cesium");

    if (isCesiumData(CesiumDataPath)) {
        auto stage = getUsdStage();
        auto cesiumData = pxr::CesiumData::Get(stage, CesiumDataPath);
        return cesiumData;
    }

    return defineCesiumData(CesiumDataPath);
}

pxr::CesiumTilesetAPI getCesiumTileset(const pxr::SdfPath& path) {
    auto stage = getUsdStage();
    auto tileset = pxr::CesiumTilesetAPI::Get(stage, path);
    assert(tileset.GetPrim().IsValid());
    return tileset;
}

pxr::CesiumRasterOverlay getCesiumRasterOverlay(const pxr::SdfPath& path) {
    auto stage = UsdUtil::getUsdStage();
    auto rasterOverlay = pxr::CesiumRasterOverlay::Get(stage, path);
    assert(rasterOverlay.GetPrim().IsValid());
    return rasterOverlay;
}

std::vector<pxr::CesiumRasterOverlay> getChildCesiumRasterOverlays(const pxr::SdfPath& path) {
    auto stage = UsdUtil::getUsdStage();
    auto prim = stage->GetPrimAtPath(path);
    assert(prim.IsValid());

    std::vector<pxr::CesiumRasterOverlay> result;

    for (const auto& childPrim : prim.GetChildren()) {
        if (childPrim.IsA<pxr::CesiumRasterOverlay>()) {
            result.emplace_back(childPrim);
        }
    }

    return result;
}

pxr::CesiumTilesetAPI getParentCesiumTileset(const pxr::SdfPath& path) {
    auto stage = UsdUtil::getUsdStage();
    auto prim = stage->GetPrimAtPath(path);
    assert(prim.IsValid());

    auto parentPath = prim.GetParent().GetPath();
    return getCesiumTileset(parentPath);
}

bool isCesiumData(const pxr::SdfPath& path) {
    auto stage = getUsdStage();
    auto prim = stage->GetPrimAtPath(path);
    if (!prim.IsValid()) {
        return false;
    }

    return prim.IsA<pxr::CesiumData>();
}

bool isCesiumTileset(const pxr::SdfPath& path) {
    auto stage = getUsdStage();
    auto prim = stage->GetPrimAtPath(path);
    if (!prim.IsValid()) {
        return false;
    }

    return prim.HasAPI<pxr::CesiumTilesetAPI>();
}

bool isCesiumRasterOverlay(const pxr::SdfPath& path) {
    auto stage = getUsdStage();
    auto prim = stage->GetPrimAtPath(path);
    if (!prim.IsValid()) {
        return false;
    }

    return prim.IsA<pxr::CesiumRasterOverlay>();
}

bool primExists(const pxr::SdfPath& path) {
    auto stage = getUsdStage();
    auto prim = stage->GetPrimAtPath(path);
    return prim.IsValid();
}

} // namespace cesium::omniverse::UsdUtil
