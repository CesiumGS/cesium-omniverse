#include "cesium/omniverse/UsdUtil.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/Tokens.h"
#include "cesium/omniverse/Viewport.h"

#include <CesiumGeometry/Transforms.h>
#include <CesiumGeospatial/Cartographic.h>
#include <CesiumGeospatial/GlobeTransforms.h>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <pxr/usd/sdf/primSpec.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/timeCode.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/xform.h>
#include <spdlog/fmt/fmt.h>

#include <regex>

namespace cesium::omniverse::UsdUtil {

namespace {
glm::dmat4 getEastNorthUpToFixedFrame(const CesiumGeospatial::Cartographic& cartographic) {
    const auto cartesian = CesiumGeospatial::Ellipsoid::WGS84.cartographicToCartesian(cartographic);
    const auto matrix = CesiumGeospatial::GlobeTransforms::eastNorthUpToFixedFrame(cartesian);
    return matrix;
}

glm::dmat4 getAxisConversionTransform() {
    const auto upAxis = getUsdUpAxis();

    auto axisConversion = glm::dmat4(1.0);

    // USD up axis can be either Y or Z
    if (upAxis == pxr::UsdGeomTokens->y) {
        axisConversion = CesiumGeometry::Transforms::Y_UP_TO_Z_UP;
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

glm::dmat4
computeUsdToEcefTransformForPrim(const CesiumGeospatial::Cartographic& origin, const pxr::SdfPath& primPath) {
    return glm::inverse(computeEcefToUsdTransformForPrim(origin, primPath));
}

Cesium3DTilesSelection::ViewState
computeViewState(const CesiumGeospatial::Cartographic& origin, const pxr::SdfPath& primPath, const Viewport& viewport) {
    const auto viewMatrix = usdToGlmMatrix(viewport.viewMatrix);
    const auto projMatrix = usdToGlmMatrix(viewport.projMatrix);
    const auto width = viewport.width;
    const auto height = viewport.height;

    const auto usdToEcef = UsdUtil::computeUsdToEcefTransformForPrim(origin, primPath);
    const auto inverseView = glm::inverse(viewMatrix);
    const auto omniCameraUp = glm::dvec3(viewMatrix[1]);
    const auto omniCameraFwd = glm::dvec3(-viewMatrix[2]);
    const auto omniCameraPosition = glm::dvec3(glm::row(inverseView, 3));
    const auto cameraUp = glm::normalize(glm::dvec3(usdToEcef * glm::dvec4(omniCameraUp, 0.0)));
    const auto cameraFwd = glm::normalize(glm::dvec3(usdToEcef * glm::dvec4(omniCameraFwd, 0.0)));
    const auto cameraPosition = glm::dvec3(usdToEcef * glm::dvec4(omniCameraPosition, 1.0));

    const auto aspect = width / height;
    const auto verticalFov = 2.0 * glm::atan(1.0 / projMatrix[1][1]);
    const auto horizontalFov = 2.0 * glm::atan(glm::tan(verticalFov * 0.5) * aspect);

    return Cesium3DTilesSelection::ViewState::create(
        cameraPosition, cameraFwd, cameraUp, glm::dvec2(width, height), horizontalFov, verticalFov);
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

    cesiumData.CreateProjectDefaultIonAccessTokenAttr();
    cesiumData.CreateProjectDefaultIonAccessTokenIdAttr();
    cesiumData.CreateDebugDisableMaterialsAttr(pxr::VtValue(false));
    cesiumData.CreateDebugDisableGeometryPoolAttr(pxr::VtValue(true));
    cesiumData.CreateDebugDisableMaterialPoolAttr(pxr::VtValue(false));
    cesiumData.CreateDebugGeometryPoolInitialCapacityAttr(pxr::VtValue(uint64_t(0)));
    cesiumData.CreateDebugMaterialPoolInitialCapacityAttr(pxr::VtValue(uint64_t(2048)));

    return cesiumData;
}

pxr::CesiumSession defineCesiumSession(const pxr::SdfPath& path) {
    auto stage = getUsdStage();
    auto cesiumSession = pxr::CesiumSession::Define(stage, path);

    cesiumSession.CreateEcefToUsdTransformAttr();

    return cesiumSession;
}

pxr::CesiumGeoreference defineCesiumGeoreference(const pxr::SdfPath& path) {
    auto stage = getUsdStage();
    auto georeference = pxr::CesiumGeoreference::Define(stage, path);

    georeference.CreateGeoreferenceOriginLatitudeAttr(pxr::VtValue(39.736401));
    georeference.CreateGeoreferenceOriginLongitudeAttr(pxr::VtValue(-105.25737));
    georeference.CreateGeoreferenceOriginHeightAttr(pxr::VtValue(2250.0));

    return georeference;
}

pxr::CesiumTilesetAPI defineCesiumTileset(const pxr::SdfPath& path) {
    auto stage = getUsdStage();
    auto mesh = pxr::UsdGeomMesh::Define(stage, path);
    mesh.GetSubdivisionSchemeAttr().Set(UsdTokens::none);
    mesh.GetInterpolateBoundaryAttr().Set(UsdTokens::none);
    assert(mesh.GetPrim().IsValid());

    auto tileset = pxr::CesiumTilesetAPI::Apply(mesh.GetPrim());
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
    tileset.CreateSmoothNormalsAttr(pxr::VtValue(false));
    tileset.CreateShowCreditsOnScreenAttr(pxr::VtValue(false));

    return tileset;
}

pxr::CesiumImagery defineCesiumImagery(const pxr::SdfPath& path) {
    auto stage = getUsdStage();
    auto imagery = pxr::CesiumImagery::Define(stage, path);
    assert(imagery.GetPrim().IsValid());

    imagery.CreateIonAssetIdAttr();
    imagery.CreateIonAccessTokenAttr();
    imagery.CreateShowCreditsOnScreenAttr(pxr::VtValue(false));

    return imagery;
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

pxr::CesiumSession getOrCreateCesiumSession() {
    static const auto CesiumSessionPath = pxr::SdfPath("/CesiumSession");

    auto stage = getUsdStage();

    if (isCesiumSession(CesiumSessionPath)) {
        auto cesiumSession = pxr::CesiumSession::Get(stage, CesiumSessionPath);
        return cesiumSession;
    }

    // Ensures that CesiumSession is created in the session layer
    const ScopedEdit scopedEdit(stage);

    // Create the CesiumSession prim
    const auto cesiumSession = defineCesiumSession(CesiumSessionPath);

    // Prevent CesiumSession from being traversed and composed into the stage
    cesiumSession.GetPrim().SetActive(false);

    return cesiumSession;
}

pxr::CesiumGeoreference getOrCreateCesiumGeoreference() {
    if (isCesiumGeoreference(GEOREFERENCE_PATH)) {
        return pxr::CesiumGeoreference::Get(getUsdStage(), GEOREFERENCE_PATH);
    }

    return defineCesiumGeoreference(GEOREFERENCE_PATH);
}

pxr::CesiumGeoreference getCesiumGeoreference(const pxr::SdfPath& path) {
    auto georeference = pxr::CesiumGeoreference::Get(getUsdStage(), path);
    assert(georeference.GetPrim().IsValid());
    return georeference;
}

pxr::CesiumTilesetAPI getCesiumTileset(const pxr::SdfPath& path) {
    auto stage = getUsdStage();
    auto tileset = pxr::CesiumTilesetAPI::Get(stage, path);
    assert(tileset.GetPrim().IsValid());
    return tileset;
}

pxr::CesiumImagery getCesiumImagery(const pxr::SdfPath& path) {
    auto stage = UsdUtil::getUsdStage();
    auto imagery = pxr::CesiumImagery::Get(stage, path);
    assert(imagery.GetPrim().IsValid());
    return imagery;
}

std::vector<pxr::CesiumImagery> getChildCesiumImageryPrims(const pxr::SdfPath& path) {
    auto stage = UsdUtil::getUsdStage();
    auto prim = stage->GetPrimAtPath(path);
    assert(prim.IsValid());

    std::vector<pxr::CesiumImagery> result;

    for (const auto& childPrim : prim.GetChildren()) {
        if (childPrim.IsA<pxr::CesiumImagery>()) {
            result.emplace_back(childPrim);
        }
    }

    return result;
}

bool isCesiumData(const pxr::SdfPath& path) {
    auto stage = getUsdStage();
    auto prim = stage->GetPrimAtPath(path);
    if (!prim.IsValid()) {
        return false;
    }

    return prim.IsA<pxr::CesiumData>();
}

bool isCesiumSession(const pxr::SdfPath& path) {
    auto stage = getUsdStage();
    auto prim = stage->GetPrimAtPath(path);
    if (!prim.IsValid()) {
        return false;
    }

    return prim.IsA<pxr::CesiumSession>();
}

bool isCesiumGeoreference(const pxr::SdfPath& path) {
    auto stage = getUsdStage();
    auto prim = stage->GetPrimAtPath(path);
    if (!prim.IsValid()) {
        return false;
    }

    return prim.IsA<pxr::CesiumGeoreference>();
}

bool isCesiumTileset(const pxr::SdfPath& path) {
    auto stage = getUsdStage();
    auto prim = stage->GetPrimAtPath(path);
    if (!prim.IsValid()) {
        return false;
    }

    return prim.HasAPI<pxr::CesiumTilesetAPI>();
}

bool isCesiumImagery(const pxr::SdfPath& path) {
    auto stage = getUsdStage();
    auto prim = stage->GetPrimAtPath(path);
    if (!prim.IsValid()) {
        return false;
    }

    return prim.IsA<pxr::CesiumImagery>();
}

bool primExists(const pxr::SdfPath& path) {
    auto stage = getUsdStage();
    auto prim = stage->GetPrimAtPath(path);
    return prim.IsValid();
}

void setGeoreferenceForTileset(const pxr::SdfPath& tilesetPath, const pxr::SdfPath& georeferencePath) {
    auto stage = getUsdStage();

    if (isCesiumTileset(tilesetPath)) {
        auto tileset = getCesiumTileset(tilesetPath);

        tileset.GetGeoreferenceBindingRel().AddTarget(georeferencePath);
    }
}

} // namespace cesium::omniverse::UsdUtil
