#include "cesium/omniverse/UsdUtil.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/GeospatialUtil.h"
#include "cesium/omniverse/Tokens.h"
#include "cesium/omniverse/Viewport.h"

#include <CesiumGeometry/Transforms.h>
#include <CesiumGeospatial/Cartographic.h>
#include <CesiumGeospatial/GlobeAnchor.h>
#include <CesiumGeospatial/GlobeTransforms.h>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <omni/ui/ImageProvider/DynamicTextureProvider.h>
#include <pxr/base/gf/rotation.h>
#include <pxr/usd/sdf/primSpec.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/timeCode.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/xformCommonAPI.h>
#include <spdlog/fmt/fmt.h>

#include <regex>

namespace cesium::omniverse::UsdUtil {

pxr::UsdStageRefPtr getUsdStage() {
    return Context::instance().getStage();
}

long getUsdStageId() {
    return Context::instance().getStageId();
}

omni::fabric::StageReaderWriter getFabricStageReaderWriter() {
    return Context::instance().getFabricStageReaderWriter();
}

omni::fabric::StageReaderWriterId getFabricStageReaderWriterId() {
    const auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    const auto stageId = getUsdStageId();
    const auto stageReaderWriterId = iStageReaderWriter->get(omni::fabric::UsdStageId{static_cast<uint64_t>(stageId)});
    return stageReaderWriterId;
}

bool hasStage() {
    return getUsdStageId() != 0;
}

glm::dvec3 usdToGlmVector(const pxr::GfVec3d& vector) {
    return {vector[0], vector[1], vector[2]};
}

glm::dmat4 usdToGlmMatrix(const pxr::GfMatrix4d& matrix) {
    // USD is row-major with left-to-right matrix multiplication
    // glm is column-major with right-to-left matrix multiplication
    // This means they have the same data layout
    return {
        matrix[0][0],
        matrix[0][1],
        matrix[0][2],
        matrix[0][3],
        matrix[1][0],
        matrix[1][1],
        matrix[1][2],
        matrix[1][3],
        matrix[2][0],
        matrix[2][1],
        matrix[2][2],
        matrix[2][3],
        matrix[3][0],
        matrix[3][1],
        matrix[3][2],
        matrix[3][3],
    };
}

pxr::GfVec3d glmToUsdVector(const glm::dvec3& vector) {
    return {vector.x, vector.y, vector.z};
}

pxr::GfVec2f glmToUsdVector(const glm::fvec2& vector) {
    return {vector.x, vector.y};
}

pxr::GfQuatd glmToUsdQuat(const glm::dquat& quat) {
    return {quat.w, quat.x, quat.y, quat.z};
}

pxr::GfVec3f glmToUsdVector(const glm::fvec3& vector) {
    return {vector.x, vector.y, vector.z};
}

pxr::GfRange3d glmToUsdRange(const std::array<glm::dvec3, 2>& extent) {
    return {glmToUsdVector(extent[0]), glmToUsdVector(extent[1])};
}

pxr::GfMatrix4d glmToUsdMatrix(const glm::dmat4& matrix) {
    // USD is row-major with left-to-right matrix multiplication
    // glm is column-major with right-to-left matrix multiplication
    // This means they have the same data layout
    return pxr::GfMatrix4d{
        matrix[0][0],
        matrix[0][1],
        matrix[0][2],
        matrix[0][3],
        matrix[1][0],
        matrix[1][1],
        matrix[1][2],
        matrix[1][3],
        matrix[2][0],
        matrix[2][1],
        matrix[2][2],
        matrix[2][3],
        matrix[3][0],
        matrix[3][1],
        matrix[3][2],
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

glm::dmat4 computeUsdLocalToWorldTransform(const pxr::SdfPath& path) {
    const auto stage = getUsdStage();
    const auto prim = stage->GetPrimAtPath(path);
    assert(prim.IsValid());
    const auto xform = pxr::UsdGeomXformable(prim);
    const auto time = pxr::UsdTimeCode::Default();
    const auto transform = xform.ComputeLocalToWorldTransform(time);
    const auto matrix = usdToGlmMatrix(transform);
    return matrix;
}

glm::dmat4 computeUsdWorldToLocalTransform(const pxr::SdfPath& path) {
    return glm::affineInverse(computeUsdLocalToWorldTransform(path));
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
    auto upAxis = pxr::UsdGeomGetStageUpAxis(stage);
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

std::string getSafeName(const std::string& name) {
    const std::regex regex("[\\W]+");
    const std::string replace = "_";

    return std::regex_replace(name, regex, replace);
}

pxr::TfToken getDynamicTextureProviderAssetPathToken(const std::string& name) {
    return pxr::TfToken(pxr::SdfAssetPath(fmt::format("dynamic://{}", name)).GetAssetPath());
}

glm::dmat4 computeEcefToUsdLocalTransform(const CesiumGeospatial::Cartographic& origin) {
    const auto cesiumDataUsd = UsdUtil::getOrCreateCesiumData();
    bool disableGeoreferencing;
    cesiumDataUsd.GetDebugDisableGeoreferencingAttr().Get(&disableGeoreferencing);

    if (disableGeoreferencing) {
        const auto scale = 1.0 / getUsdMetersPerUnit();
        return glm::scale(glm::dmat4(1.0), glm::dvec3(scale));
    }

    return GeospatialUtil::getCoordinateSystem(origin).getEcefToLocalTransformation();
}

glm::dmat4
computeEcefToUsdWorldTransformForPrim(const CesiumGeospatial::Cartographic& origin, const pxr::SdfPath& primPath) {
    const auto ecefToUsdTransform = computeEcefToUsdLocalTransform(origin);
    const auto primUsdWorldTransform = computeUsdLocalToWorldTransform(primPath);
    const auto primEcefToUsdTransform = primUsdWorldTransform * ecefToUsdTransform;
    return primEcefToUsdTransform;
}

glm::dmat4
computeUsdWorldToEcefTransformForPrim(const CesiumGeospatial::Cartographic& origin, const pxr::SdfPath& primPath) {
    return glm::affineInverse(computeEcefToUsdWorldTransformForPrim(origin, primPath));
}

glm::dmat4
computeEcefToUsdLocalTransformForPrim(const CesiumGeospatial::Cartographic& origin, const pxr::SdfPath& primPath) {
    const auto ecefToUsdTransform = computeEcefToUsdLocalTransform(origin);
    const auto usdWorldToLocalTransform = UsdUtil::computeUsdWorldToLocalTransform(primPath);
    const auto primEcefToUsdTransform = usdWorldToLocalTransform * ecefToUsdTransform;
    return primEcefToUsdTransform;
}

glm::dmat4
computeUsdLocalToEcefTransformForPrim(const CesiumGeospatial::Cartographic& origin, const pxr::SdfPath& primPath) {
    return glm::affineInverse(computeEcefToUsdLocalTransformForPrim(origin, primPath));
}

Cesium3DTilesSelection::ViewState
computeViewState(const CesiumGeospatial::Cartographic& origin, const pxr::SdfPath& primPath, const Viewport& viewport) {
    const auto viewMatrix = usdToGlmMatrix(viewport.viewMatrix);
    const auto projMatrix = usdToGlmMatrix(viewport.projMatrix);
    const auto width = viewport.width;
    const auto height = viewport.height;

    const auto usdToEcef = UsdUtil::computeUsdWorldToEcefTransformForPrim(origin, primPath);
    const auto inverseView = glm::inverse(viewMatrix);
    const auto omniCameraUp = glm::dvec3(inverseView[1]);
    const auto omniCameraFwd = glm::dvec3(-inverseView[2]);
    const auto omniCameraPosition = glm::dvec3(inverseView[3]);
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

    return {glmToUsdVector(worldMin), glmToUsdVector(worldMax)};
}

pxr::GfVec3f getEulerAnglesFromQuaternion(const pxr::GfQuatf& quaternion) {
    const auto rotation = pxr::GfRotation(quaternion);
    const auto euler = rotation.Decompose(pxr::GfVec3d::XAxis(), pxr::GfVec3d::YAxis(), pxr::GfVec3d::ZAxis());
    return {static_cast<float>(euler[0]), static_cast<float>(euler[1]), static_cast<float>(euler[2])};
}

pxr::CesiumData defineCesiumData(const pxr::SdfPath& path) {
    auto stage = getUsdStage();
    auto cesiumData = pxr::CesiumData::Define(stage, path);

    return cesiumData;
}

pxr::CesiumSession defineCesiumSession(const pxr::SdfPath& path) {
    auto stage = getUsdStage();
    auto cesiumSession = pxr::CesiumSession::Define(stage, path);

    return cesiumSession;
}

pxr::CesiumGeoreference defineCesiumGeoreference(const pxr::SdfPath& path) {
    auto stage = getUsdStage();
    auto georeference = pxr::CesiumGeoreference::Define(stage, path);

    return georeference;
}

pxr::CesiumTileset defineCesiumTileset(const pxr::SdfPath& path) {
    auto stage = getUsdStage();
    auto tileset = pxr::CesiumTileset::Define(stage, path);

    assert(tileset.GetPrim().IsValid());
    return tileset;
}

pxr::CesiumImagery defineCesiumImagery(const pxr::SdfPath& path) {
    auto stage = getUsdStage();
    auto imagery = pxr::CesiumImagery::Define(stage, path);
    assert(imagery.GetPrim().IsValid());

    return imagery;
}

pxr::CesiumGlobeAnchorAPI defineGlobeAnchor(const pxr::SdfPath& path) {
    auto stage = getUsdStage();
    auto prim = stage->GetPrimAtPath(path);

    auto globeAnchor = pxr::CesiumGlobeAnchorAPI::Apply(prim);
    assert(globeAnchor.GetPrim().IsValid());

    return globeAnchor;
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

pxr::CesiumTileset getCesiumTileset(const pxr::SdfPath& path) {
    auto stage = getUsdStage();
    auto tileset = pxr::CesiumTileset::Get(stage, path);
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

pxr::CesiumGlobeAnchorAPI getCesiumGlobeAnchor(const pxr::SdfPath& path) {
    auto stage = UsdUtil::getUsdStage();
    auto globeAnchor = pxr::CesiumGlobeAnchorAPI::Get(stage, path);
    assert(globeAnchor.GetPrim().IsValid());
    return globeAnchor;
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

    return prim.IsA<pxr::CesiumTileset>();
}

bool isCesiumImagery(const pxr::SdfPath& path) {
    auto stage = getUsdStage();
    auto prim = stage->GetPrimAtPath(path);
    if (!prim.IsValid()) {
        return false;
    }

    return prim.IsA<pxr::CesiumImagery>();
}

bool hasCesiumGlobeAnchor(const pxr::SdfPath& path) {
    auto stage = getUsdStage();
    auto prim = stage->GetPrimAtPath(path);
    if (!prim.IsValid()) {
        return false;
    }

    return prim.HasAPI<pxr::CesiumGlobeAnchorAPI>();
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

void addOrUpdateTransformOpForAnchor(const pxr::SdfPath& path, const glm::dmat4& transform) {
    auto prim = getUsdStage()->GetPrimAtPath(path);

    if (!hasCesiumGlobeAnchor(path)) {
        return;
    }

    auto xform = pxr::UsdGeomXform(prim);
    auto resetXformStack = xform.GetResetXformStack();
    auto xformOps = xform.GetOrderedXformOps(&resetXformStack);

    auto hasCesiumSuffix = [](auto op) { return op.HasSuffix(pxr::UsdTokens->cesium); };
    auto transformOp = std::find_if(xformOps.begin(), xformOps.end(), hasCesiumSuffix);

    if (transformOp != xformOps.end()) {
        transformOp->Set(UsdUtil::glmToUsdMatrix(transform));
    } else {
        // We need to do this when it's a new anchor.
        xform.ClearXformOpOrder();

        // We reset the TRS values to defaults, so they can be used as an offset.
        //   This has the side effect of "baking" the transform at the time that the anchor is added
        //   to the into the anchor transform. Maybe we can fix that later.
        auto xformCommonApi = pxr::UsdGeomXformCommonAPI(prim);
        xformCommonApi.SetTranslate(pxr::GfVec3d(0., 0., 0.));
        xformCommonApi.SetRotate(pxr::GfVec3f(0.f, 0.f, 0.f));
        xformCommonApi.SetScale(pxr::GfVec3f(1.f, 1.f, 1.f));

        xform.AddTransformOp(pxr::UsdGeomXformOp::PrecisionDouble, pxr::UsdTokens->cesium)
            .Set(UsdUtil::glmToUsdMatrix(transform));
    }
}

std::optional<pxr::GfMatrix4d> getCesiumTransformOpValueForPathIfExists(const pxr::SdfPath& path) {
    auto prim = getUsdStage()->GetPrimAtPath(path);
    auto xform = pxr::UsdGeomXform(prim);
    auto resetXformStack = xform.GetResetXformStack();
    auto xformOps = xform.GetOrderedXformOps(&resetXformStack);

    auto hasCesiumSuffix = [](auto op) { return op.HasSuffix(pxr::UsdTokens->cesium); };
    auto transformOp = std::find_if(xformOps.begin(), xformOps.end(), hasCesiumSuffix);

    if (transformOp != xformOps.end()) {
        pxr::GfMatrix4d transform;
        transformOp->Get(&transform);

        return transform;
    }

    return std::nullopt;
}

std::optional<pxr::SdfPath> getAnchorGeoreferencePath(const pxr::SdfPath& path) {
    if (!hasCesiumGlobeAnchor(path)) {
        return std::nullopt;
    }

    auto globeAnchor = getCesiumGlobeAnchor(path);
    pxr::SdfPathVector targets;
    if (!globeAnchor.GetGeoreferenceBindingRel().GetForwardedTargets(&targets)) {
        return std::nullopt;
    }

    return targets[0];
}

std::optional<CesiumGeospatial::Cartographic> getCartographicOriginForAnchor(const pxr::SdfPath& path) {
    auto anchorGeoreferencePath = getAnchorGeoreferencePath(path);

    if (!anchorGeoreferencePath.has_value()) {
        return std::nullopt;
    }

    auto georeferenceOrigin = UsdUtil::getCesiumGeoreference(anchorGeoreferencePath.value());
    return GeospatialUtil::convertGeoreferenceToCartographic(georeferenceOrigin);
}

} // namespace cesium::omniverse::UsdUtil
