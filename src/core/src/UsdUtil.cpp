#include "cesium/omniverse/UsdUtil.h"

#include "CesiumUsdSchemas/cartographicPolygon.h"

#include "cesium/omniverse/AssetRegistry.h"
#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/MathUtil.h"
#include "cesium/omniverse/OmniData.h"
#include "cesium/omniverse/OmniGeoreference.h"
#include "cesium/omniverse/UsdTokens.h"
#include "cesium/omniverse/Viewport.h"

#include <Cesium3DTilesSelection/ViewState.h>
#include <CesiumGeometry/Transforms.h>
#include <CesiumGeospatial/Cartographic.h>
#include <CesiumGeospatial/Ellipsoid.h>
#include <CesiumGeospatial/GlobeAnchor.h>
#include <CesiumGeospatial/GlobeTransforms.h>
#include <CesiumGeospatial/LocalHorizontalCoordinateSystem.h>
#include <CesiumUsdSchemas/cartographicPolygon.h>
#include <CesiumUsdSchemas/data.h>
#include <CesiumUsdSchemas/georeference.h>
#include <CesiumUsdSchemas/globeAnchorAPI.h>
#include <CesiumUsdSchemas/ionImagery.h>
#include <CesiumUsdSchemas/ionServer.h>
#include <CesiumUsdSchemas/polygonImagery.h>
#include <CesiumUsdSchemas/session.h>
#include <CesiumUsdSchemas/tileset.h>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <omni/ui/ImageProvider/DynamicTextureProvider.h>
#include <pxr/base/gf/rotation.h>
#include <pxr/usd/sdf/primSpec.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/timeCode.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/xformCommonAPI.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/shader.h>
#include <spdlog/fmt/fmt.h>

#include <cctype>

namespace cesium::omniverse::UsdUtil {

namespace {

class ScopedEdit {
  public:
    ScopedEdit(const PXR_NS::UsdStageWeakPtr& pStage)
        : _pStage(pStage)
        , _sessionLayer(_pStage->GetSessionLayer())
        , _sessionLayerWasEditable(_sessionLayer->PermissionToEdit())
        , _originalEditTarget(_pStage->GetEditTarget()) {

        _sessionLayer->SetPermissionToEdit(true);
        _pStage->SetEditTarget(PXR_NS::UsdEditTarget(_sessionLayer));
    }
    ~ScopedEdit() {
        _sessionLayer->SetPermissionToEdit(_sessionLayerWasEditable);
        _pStage->SetEditTarget(_originalEditTarget);
    }
    ScopedEdit(const ScopedEdit&) = delete;
    ScopedEdit& operator=(const ScopedEdit&) = delete;
    ScopedEdit(ScopedEdit&&) noexcept = delete;
    ScopedEdit& operator=(ScopedEdit&&) noexcept = delete;

  private:
    PXR_NS::UsdStageWeakPtr _pStage;
    PXR_NS::SdfLayerHandle _sessionLayer;
    bool _sessionLayerWasEditable;
    PXR_NS::UsdEditTarget _originalEditTarget;
};

bool getDebugDisableGeoreferencing(const Context& context) {
    const auto pData = context.getAssetRegistry().getFirstData();
    if (!pData) {
        return false;
    }

    return pData->getDebugDisableGeoreferencing();
}

} // namespace

glm::dvec3 usdToGlmVector(const PXR_NS::GfVec3d& vector) {
    return {vector[0], vector[1], vector[2]};
}

glm::fvec3 usdToGlmVector(const PXR_NS::GfVec3f& vector) {
    return {vector[0], vector[1], vector[2]};
}

glm::dmat4 usdToGlmMatrix(const PXR_NS::GfMatrix4d& matrix) {
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

std::array<glm::dvec3, 2> usdToGlmExtent(const PXR_NS::GfRange3d& extent) {
    return {{usdToGlmVector(extent.GetMin()), usdToGlmVector(extent.GetMax())}};
}

PXR_NS::GfVec3d glmToUsdVector(const glm::dvec3& vector) {
    return {vector.x, vector.y, vector.z};
}

PXR_NS::GfVec2f glmToUsdVector(const glm::fvec2& vector) {
    return {vector.x, vector.y};
}

PXR_NS::GfVec3f glmToUsdVector(const glm::fvec3& vector) {
    return {vector.x, vector.y, vector.z};
}

PXR_NS::GfVec4f glmToUsdVector(const glm::fvec4& vector) {
    return {vector.x, vector.y, vector.z, vector.w};
}

PXR_NS::GfRange3d glmToUsdExtent(const std::array<glm::dvec3, 2>& extent) {
    return {glmToUsdVector(extent[0]), glmToUsdVector(extent[1])};
}

PXR_NS::GfQuatd glmToUsdQuat(const glm::dquat& quat) {
    return {quat.w, quat.x, quat.y, quat.z};
}

PXR_NS::GfQuatf glmToUsdQuat(const glm::fquat& quat) {
    return {quat.w, quat.x, quat.y, quat.z};
}

PXR_NS::GfMatrix4d glmToUsdMatrix(const glm::dmat4& matrix) {
    // USD is row-major with left-to-right matrix multiplication
    // glm is column-major with right-to-left matrix multiplication
    // This means they have the same data layout
    return PXR_NS::GfMatrix4d{
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

glm::dmat4 computePrimLocalToWorldTransform(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path) {
    const auto prim = pStage->GetPrimAtPath(path);

    if (!prim.IsA<PXR_NS::UsdGeomXformable>()) {
        return glm::dmat4(1.0);
    }

    const auto xform = PXR_NS::UsdGeomXformable(prim);
    const auto time = PXR_NS::UsdTimeCode::Default();
    const auto transform = xform.ComputeLocalToWorldTransform(time);
    return usdToGlmMatrix(transform);
}

glm::dmat4 computePrimWorldToLocalTransform(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path) {
    return glm::affineInverse(computePrimLocalToWorldTransform(pStage, path));
}

glm::dmat4 computeEcefToStageTransform(const Context& context, const PXR_NS::SdfPath& georeferencePath) {
    const auto disableGeoreferencing = getDebugDisableGeoreferencing(context);
    const auto pGeoreference = context.getAssetRegistry().getGeoreference(georeferencePath);

    if (disableGeoreferencing || !pGeoreference) {
        const auto zUp = getUsdUpAxis(context.getUsdStage()) == PXR_NS::UsdGeomTokens->z;
        const auto axisConversion = zUp ? glm::dmat4(1.0) : CesiumGeometry::Transforms::Z_UP_TO_Y_UP;
        const auto scale = glm::scale(glm::dmat4(1.0), glm::dvec3(1.0 / getUsdMetersPerUnit(context.getUsdStage())));
        return scale * axisConversion;
    }

    return pGeoreference->getLocalCoordinateSystem().getEcefToLocalTransformation();
}

glm::dmat4 computeEcefToPrimWorldTransform(
    const Context& context,
    const PXR_NS::SdfPath& georeferencePath,
    const PXR_NS::SdfPath& primPath) {
    const auto ecefToStageTransform = computeEcefToStageTransform(context, georeferencePath);
    const auto primLocalToWorldTransform = computePrimLocalToWorldTransform(context.getUsdStage(), primPath);
    return primLocalToWorldTransform * ecefToStageTransform;
}

glm::dmat4 computePrimWorldToEcefTransform(
    const Context& context,
    const PXR_NS::SdfPath& georeferencePath,
    const PXR_NS::SdfPath& primPath) {
    return glm::affineInverse(computeEcefToPrimWorldTransform(context, georeferencePath, primPath));
}

glm::dmat4 computeEcefToPrimLocalTransform(
    const Context& context,
    const PXR_NS::SdfPath& georeferencePath,
    const PXR_NS::SdfPath& primPath) {
    const auto ecefToStageTransform = computeEcefToStageTransform(context, georeferencePath);
    const auto primWorldToLocalTransform = computePrimWorldToLocalTransform(context.getUsdStage(), primPath);
    return primWorldToLocalTransform * ecefToStageTransform;
}

glm::dmat4 computePrimLocalToEcefTransform(
    const Context& context,
    const PXR_NS::SdfPath& georeferencePath,
    const PXR_NS::SdfPath& primPath) {
    return glm::affineInverse(computeEcefToPrimLocalTransform(context, georeferencePath, primPath));
}

Cesium3DTilesSelection::ViewState computeViewState(
    const Context& context,
    const PXR_NS::SdfPath& georeferencePath,
    const PXR_NS::SdfPath& primPath,
    const Viewport& viewport) {
    const auto& viewMatrix = viewport.viewMatrix;
    const auto& projMatrix = viewport.projMatrix;
    const auto width = viewport.width;
    const auto height = viewport.height;

    const auto primWorldToEcefTransform = computePrimWorldToEcefTransform(context, georeferencePath, primPath);
    const auto inverseView = glm::affineInverse(viewMatrix);
    const auto usdCameraUp = glm::dvec3(inverseView[1]);
    const auto usdCameraFwd = glm::dvec3(-inverseView[2]);
    const auto usdCameraPosition = glm::dvec3(inverseView[3]);
    const auto cameraUp = glm::normalize(glm::dvec3(primWorldToEcefTransform * glm::dvec4(usdCameraUp, 0.0)));
    const auto cameraFwd = glm::normalize(glm::dvec3(primWorldToEcefTransform * glm::dvec4(usdCameraFwd, 0.0)));
    const auto cameraPosition = glm::dvec3(primWorldToEcefTransform * glm::dvec4(usdCameraPosition, 1.0));

    const auto aspect = width / height;
    const auto verticalFov = 2.0 * glm::atan(1.0 / projMatrix[1][1]);
    const auto horizontalFov = 2.0 * glm::atan(glm::tan(verticalFov * 0.5) * aspect);

    return Cesium3DTilesSelection::ViewState::create(
        cameraPosition, cameraFwd, cameraUp, glm::dvec2(width, height), horizontalFov, verticalFov);
}

bool primExists(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path) {
    return pStage->GetPrimAtPath(path).IsValid();
}

bool isPrimVisible(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path) {
    // This is similar to isPrimVisible in kit-sdk/dev/include/omni/usd/UsdUtils.h
    const auto prim = pStage->GetPrimAtPath(path);

    if (!prim.IsA<PXR_NS::UsdGeomImageable>()) {
        return false;
    }

    const auto imageable = PXR_NS::UsdGeomImageable(prim);
    const auto time = PXR_NS::UsdTimeCode::Default();
    const auto visibility = imageable.ComputeVisibility(time);
    return visibility != PXR_NS::UsdGeomTokens->invisible;
}

const std::string& getName(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path) {
    return pStage->GetPrimAtPath(path).GetName().GetString();
}

PXR_NS::TfToken getUsdUpAxis(const PXR_NS::UsdStageWeakPtr& pStage) {
    return PXR_NS::UsdGeomGetStageUpAxis(pStage);
}

double getUsdMetersPerUnit(const PXR_NS::UsdStageWeakPtr& pStage) {
    return PXR_NS::UsdGeomGetStageMetersPerUnit(pStage);
}

PXR_NS::SdfPath getRootPath(const PXR_NS::UsdStageWeakPtr& pStage) {
    return pStage->GetPseudoRoot().GetPath();
}

PXR_NS::SdfPath
makeUniquePath(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& parentPath, const std::string& name) {
    PXR_NS::UsdPrim prim;
    PXR_NS::SdfPath path;
    auto copy = 0;

    do {
        const auto copyName = copy > 0 ? fmt::format("{}_{}", name, copy) : name;
        path = parentPath.AppendChild(PXR_NS::TfToken(copyName));
        prim = pStage->GetPrimAtPath(path);
        ++copy;
    } while (prim.IsValid());

    return path;
}

std::string getSafeName(const std::string& name) {
    auto safeName = name;

    for (auto& c : safeName) {
        if (!std::isalnum(c)) {
            c = '_';
        }
    }

    return safeName;
}

PXR_NS::TfToken getDynamicTextureProviderAssetPathToken(const std::string_view& name) {
    return PXR_NS::TfToken(PXR_NS::SdfAssetPath(fmt::format("dynamic://{}", name)).GetAssetPath());
}

PXR_NS::CesiumData defineCesiumData(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path) {
    return PXR_NS::CesiumData::Define(pStage, path);
}

PXR_NS::CesiumTileset defineCesiumTileset(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path) {
    return PXR_NS::CesiumTileset::Define(pStage, path);
}

PXR_NS::CesiumIonImagery defineCesiumIonImagery(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path) {
    return PXR_NS::CesiumIonImagery::Define(pStage, path);
}

PXR_NS::CesiumPolygonImagery
defineCesiumPolygonImagery(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path) {
    return PXR_NS::CesiumPolygonImagery::Define(pStage, path);
}

PXR_NS::CesiumGeoreference
defineCesiumGeoreference(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path) {
    return PXR_NS::CesiumGeoreference::Define(pStage, path);
}

PXR_NS::CesiumIonServer defineCesiumIonServer(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path) {
    return PXR_NS::CesiumIonServer::Define(pStage, path);
}

PXR_NS::CesiumGlobeAnchorAPI
applyCesiumGlobeAnchor(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path) {
    const auto prim = pStage->GetPrimAtPath(path);
    assert(prim.IsValid() && prim.IsA<PXR_NS::UsdGeomXformable>());
    return PXR_NS::CesiumGlobeAnchorAPI::Apply(prim);
}

PXR_NS::CesiumSession defineCesiumSession(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path) {
    return PXR_NS::CesiumSession::Define(pStage, path);
}

PXR_NS::CesiumData getCesiumData(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path) {
    return PXR_NS::CesiumData::Get(pStage, path);
}

PXR_NS::CesiumTileset getCesiumTileset(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path) {
    return PXR_NS::CesiumTileset::Get(pStage, path);
}

PXR_NS::CesiumImagery getCesiumImagery(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path) {
    return PXR_NS::CesiumImagery::Get(pStage, path);
}

PXR_NS::CesiumIonImagery getCesiumIonImagery(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path) {
    return PXR_NS::CesiumIonImagery::Get(pStage, path);
}

PXR_NS::CesiumPolygonImagery
getCesiumPolygonImagery(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path) {
    return PXR_NS::CesiumPolygonImagery::Get(pStage, path);
}

PXR_NS::CesiumGeoreference getCesiumGeoreference(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path) {
    return PXR_NS::CesiumGeoreference::Get(pStage, path);
}

PXR_NS::CesiumGlobeAnchorAPI getCesiumGlobeAnchor(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path) {
    return PXR_NS::CesiumGlobeAnchorAPI::Get(pStage, path);
}

PXR_NS::CesiumIonServer getCesiumIonServer(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path) {
    return PXR_NS::CesiumIonServer::Get(pStage, path);
}

PXR_NS::CesiumCartographicPolygon
getCesiumCartographicPolygon(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path) {
    return PXR_NS::CesiumCartographicPolygon::Get(pStage, path);
}

PXR_NS::UsdShadeShader getUsdShader(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path) {
    return PXR_NS::UsdShadeShader::Get(pStage, path);
}

PXR_NS::UsdGeomBasisCurves getUsdBasisCurves(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path) {
    return PXR_NS::UsdGeomBasisCurves::Get(pStage, path);
}

PXR_NS::CesiumSession getOrCreateCesiumSession(const PXR_NS::UsdStageWeakPtr& pStage) {
    static const auto CesiumSessionPath = PXR_NS::SdfPath("/CesiumSession");

    if (isCesiumSession(pStage, CesiumSessionPath)) {
        return PXR_NS::CesiumSession::Get(pStage, CesiumSessionPath);
    }

    // Ensures that CesiumSession is created in the session layer
    const ScopedEdit scopedEdit(pStage);

    // Create the CesiumSession
    const auto cesiumSession = defineCesiumSession(pStage, CesiumSessionPath);

    // Prevent CesiumSession from being traversed and composed into the stage
    cesiumSession.GetPrim().SetActive(false);

    return cesiumSession;
}

bool isCesiumData(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path) {
    const auto prim = pStage->GetPrimAtPath(path);
    if (!prim.IsValid()) {
        return false;
    }

    return prim.IsA<PXR_NS::CesiumData>();
}

bool isCesiumTileset(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path) {
    const auto prim = pStage->GetPrimAtPath(path);
    if (!prim.IsValid()) {
        return false;
    }

    return prim.IsA<PXR_NS::CesiumTileset>();
}

bool isCesiumImagery(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path) {
    const auto prim = pStage->GetPrimAtPath(path);
    if (!prim.IsValid()) {
        return false;
    }

    return prim.IsA<PXR_NS::CesiumImagery>();
}

bool isCesiumIonImagery(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path) {
    const auto prim = pStage->GetPrimAtPath(path);
    if (!prim.IsValid()) {
        return false;
    }

    return prim.IsA<PXR_NS::CesiumIonImagery>();
}

bool isCesiumPolygonImagery(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path) {
    const auto prim = pStage->GetPrimAtPath(path);
    if (!prim.IsValid()) {
        return false;
    }

    return prim.IsA<PXR_NS::CesiumPolygonImagery>();
}

bool isCesiumGeoreference(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path) {
    const auto prim = pStage->GetPrimAtPath(path);
    if (!prim.IsValid()) {
        return false;
    }

    return prim.IsA<PXR_NS::CesiumGeoreference>();
}

bool isCesiumIonServer(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path) {
    const auto prim = pStage->GetPrimAtPath(path);
    if (!prim.IsValid()) {
        return false;
    }

    return prim.IsA<PXR_NS::CesiumIonServer>();
}

bool isCesiumCartographicPolygon(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path) {
    const auto prim = pStage->GetPrimAtPath(path);
    if (!prim.IsValid()) {
        return false;
    }

    return prim.IsA<PXR_NS::CesiumCartographicPolygon>();
}

bool isCesiumSession(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path) {
    const auto prim = pStage->GetPrimAtPath(path);
    if (!prim.IsValid()) {
        return false;
    }

    return prim.IsA<PXR_NS::CesiumSession>();
}

bool hasCesiumGlobeAnchor(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path) {
    const auto prim = pStage->GetPrimAtPath(path);
    if (!prim.IsValid()) {
        return false;
    }

    return prim.IsA<PXR_NS::UsdGeomXformable>() && prim.HasAPI<PXR_NS::CesiumGlobeAnchorAPI>();
}

bool isUsdShader(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path) {
    const auto prim = pStage->GetPrimAtPath(path);
    if (!prim.IsValid()) {
        return false;
    }

    return prim.IsA<PXR_NS::UsdShadeShader>();
}

bool isUsdMaterial(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path) {
    const auto prim = pStage->GetPrimAtPath(path);
    if (!prim.IsValid()) {
        return false;
    }

    return prim.IsA<PXR_NS::UsdShadeMaterial>();
}

std::optional<TranslateRotateScaleOps> getTranslateRotateScaleOps(const PXR_NS::UsdGeomXformable& xformable) {
    const PXR_NS::UsdGeomXformOp* pTranslateOp = nullptr;
    const PXR_NS::UsdGeomXformOp* pRotateOp = nullptr;
    const PXR_NS::UsdGeomXformOp* pScaleOp = nullptr;

    auto translateOpIndex = 0;
    auto rotateOpIndex = 0;
    auto scaleOpIndex = 0;

    auto opIndex = 0;

    bool resetsXformStack;
    const auto xformOps = xformable.GetOrderedXformOps(&resetsXformStack);
    auto eulerAngleOrder = MathUtil::EulerAngleOrder::XYZ;

    for (const auto& xformOp : xformOps) {
        switch (xformOp.GetOpType()) {
            case PXR_NS::UsdGeomXformOp::TypeTranslate:
                pTranslateOp = &xformOp;
                translateOpIndex = opIndex;
                break;
            case PXR_NS::UsdGeomXformOp::TypeRotateXYZ:
                eulerAngleOrder = MathUtil::EulerAngleOrder::XYZ;
                pRotateOp = &xformOp;
                rotateOpIndex = opIndex;
                break;
            case PXR_NS::UsdGeomXformOp::TypeRotateXZY:
                eulerAngleOrder = MathUtil::EulerAngleOrder::XZY;
                pRotateOp = &xformOp;
                rotateOpIndex = opIndex;
                break;
            case PXR_NS::UsdGeomXformOp::TypeRotateYXZ:
                eulerAngleOrder = MathUtil::EulerAngleOrder::YXZ;
                pRotateOp = &xformOp;
                rotateOpIndex = opIndex;
                break;
            case PXR_NS::UsdGeomXformOp::TypeRotateYZX:
                eulerAngleOrder = MathUtil::EulerAngleOrder::YZX;
                pRotateOp = &xformOp;
                rotateOpIndex = opIndex;
                break;
            case PXR_NS::UsdGeomXformOp::TypeRotateZXY:
                eulerAngleOrder = MathUtil::EulerAngleOrder::ZXY;
                pRotateOp = &xformOp;
                rotateOpIndex = opIndex;
                break;
            case PXR_NS::UsdGeomXformOp::TypeRotateZYX:
                eulerAngleOrder = MathUtil::EulerAngleOrder::ZYX;
                pRotateOp = &xformOp;
                rotateOpIndex = opIndex;
                break;
            case PXR_NS::UsdGeomXformOp::TypeScale:
                pScaleOp = &xformOp;
                scaleOpIndex = opIndex;
                break;
            default:
                break;
        }
        ++opIndex;
    }

    if (!pTranslateOp || !pRotateOp || !pScaleOp) {
        return std::nullopt;
    }

    if (translateOpIndex != 0 || rotateOpIndex != 1 || scaleOpIndex != 2) {
        return std::nullopt;
    }

    if (opIndex != 3) {
        return std::nullopt;
    }

    const auto isPrecisionSupported = [](PXR_NS::UsdGeomXformOp::Precision precision) {
        return precision == PXR_NS::UsdGeomXformOp::PrecisionDouble ||
               precision == PXR_NS::UsdGeomXformOp::PrecisionFloat;
    };

    const auto translatePrecisionSupported = isPrecisionSupported(pTranslateOp->GetPrecision());
    const auto rotatePrecisionSupported = isPrecisionSupported(pRotateOp->GetPrecision());
    const auto scalePrecisionSupported = isPrecisionSupported(pScaleOp->GetPrecision());

    if (!translatePrecisionSupported || !rotatePrecisionSupported || !scalePrecisionSupported) {
        return std::nullopt;
    }

    return TranslateRotateScaleOps{pTranslateOp, pRotateOp, pScaleOp, eulerAngleOrder};
}

} // namespace cesium::omniverse::UsdUtil
