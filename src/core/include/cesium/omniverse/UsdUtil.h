#pragma once

#include <glm/fwd.hpp>
#include <pxr/base/gf/declare.h>
#include <pxr/usd/usd/common.h>

PXR_NAMESPACE_OPEN_SCOPE
class CesiumCartographicPolygon;
class CesiumData;
class CesiumGeoreference;
class CesiumGlobeAnchorAPI;
class CesiumImagery;
class CesiumIonImagery;
class CesiumIonServer;
class CesiumPolygonImagery;
class CesiumSession;
class CesiumTileset;
class UsdGeomBasisCurves;
class UsdGeomXformable;
class UsdGeomXformOp;
class UsdShadeShader;
PXR_NAMESPACE_CLOSE_SCOPE

namespace Cesium3DTilesSelection {
class ViewState;
}

namespace CesiumGeospatial {
class Cartographic;
class Ellipsoid;
class LocalHorizontalCoordinateSystem;
} // namespace CesiumGeospatial

namespace cesium::omniverse {
class Context;
struct Viewport;
} // namespace cesium::omniverse

namespace cesium::omniverse::MathUtil {
enum class EulerAngleOrder;
}

namespace cesium::omniverse::UsdUtil {

glm::dvec3 usdToGlmVector(const PXR_NS::GfVec3d& vector);
glm::fvec3 usdToGlmVector(const PXR_NS::GfVec3f& vector);
glm::dmat4 usdToGlmMatrix(const PXR_NS::GfMatrix4d& matrix);
std::array<glm::dvec3, 2> usdToGlmExtent(const PXR_NS::GfRange3d& extent);

PXR_NS::GfVec3d glmToUsdVector(const glm::dvec3& vector);
PXR_NS::GfVec2f glmToUsdVector(const glm::fvec2& vector);
PXR_NS::GfVec3f glmToUsdVector(const glm::fvec3& vector);
PXR_NS::GfVec4f glmToUsdVector(const glm::fvec4& vector);
PXR_NS::GfRange3d glmToUsdExtent(const std::array<glm::dvec3, 2>& extent);
PXR_NS::GfQuatd glmToUsdQuat(const glm::dquat& quat);
PXR_NS::GfQuatf glmToUsdQuat(const glm::fquat& quat);
PXR_NS::GfMatrix4d glmToUsdMatrix(const glm::dmat4& matrix);

glm::dmat4 computePrimLocalToWorldTransform(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path);
glm::dmat4 computePrimWorldToLocalTransform(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path);
glm::dmat4 computeEcefToStageTransform(Context* pContext, const PXR_NS::SdfPath& georeferencePath);
glm::dmat4 computeEcefToPrimWorldTransform(
    Context* pContext,
    const PXR_NS::SdfPath& georeferencePath,
    const PXR_NS::SdfPath& primPath);
glm::dmat4 computePrimWorldToEcefTransform(
    Context* pContext,
    const PXR_NS::SdfPath& georeferencePath,
    const PXR_NS::SdfPath& primPath);
glm::dmat4 computeEcefToPrimLocalTransform(
    Context* pContext,
    const PXR_NS::SdfPath& georeferencePath,
    const PXR_NS::SdfPath& primPath);
glm::dmat4 computePrimLocalToEcefTransform(
    Context* pContext,
    const PXR_NS::SdfPath& georeferencePath,
    const PXR_NS::SdfPath& primPath);
CesiumGeospatial::LocalHorizontalCoordinateSystem computeLocalCoordinateSystem(
    const PXR_NS::UsdStageWeakPtr& pStage,
    const CesiumGeospatial::Cartographic& origin,
    const CesiumGeospatial::Ellipsoid& ellipsoid);

Cesium3DTilesSelection::ViewState computeViewState(
    Context* pContext,
    const PXR_NS::SdfPath& georeferencePath,
    const PXR_NS::SdfPath& primPath,
    const Viewport& viewport);

bool primExists(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path);
bool isPrimVisible(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path);
const std::string& getName(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path);

PXR_NS::TfToken getUsdUpAxis(const PXR_NS::UsdStageWeakPtr& pStage);
double getUsdMetersPerUnit(const PXR_NS::UsdStageWeakPtr& pStage);
PXR_NS::SdfPath getRootPath(const PXR_NS::UsdStageWeakPtr& pStage);
PXR_NS::SdfPath
makeUniquePath(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& parentPath, const std::string& name);
std::string getSafeName(const std::string& name);
PXR_NS::TfToken getDynamicTextureProviderAssetPathToken(const std::string_view& name);

PXR_NS::CesiumData defineCesiumData(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path);
PXR_NS::CesiumTileset defineCesiumTileset(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path);
PXR_NS::CesiumIonImagery defineCesiumIonImagery(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path);
PXR_NS::CesiumPolygonImagery
defineCesiumPolygonImagery(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path);
PXR_NS::CesiumGeoreference defineCesiumGeoreference(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path);
PXR_NS::CesiumIonServer defineCesiumIonServer(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path);
PXR_NS::CesiumGlobeAnchorAPI applyCesiumGlobeAnchor(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path);
PXR_NS::CesiumSession defineCesiumSession(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path);

PXR_NS::CesiumData getCesiumData(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path);
PXR_NS::CesiumTileset getCesiumTileset(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path);
PXR_NS::CesiumImagery getCesiumImagery(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path);
PXR_NS::CesiumIonImagery getCesiumIonImagery(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path);
PXR_NS::CesiumPolygonImagery
getCesiumPolygonImagery(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path);
PXR_NS::CesiumGeoreference getCesiumGeoreference(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path);
PXR_NS::CesiumGlobeAnchorAPI getCesiumGlobeAnchor(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path);
PXR_NS::CesiumIonServer getCesiumIonServer(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path);
PXR_NS::CesiumCartographicPolygon
getCesiumCartographicPolygon(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path);
PXR_NS::UsdShadeShader getUsdShader(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path);
PXR_NS::UsdGeomBasisCurves getUsdBasisCurves(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path);

PXR_NS::CesiumSession getOrCreateCesiumSession(const PXR_NS::UsdStageWeakPtr& pStage);

bool isCesiumData(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path);
bool isCesiumTileset(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path);
bool isCesiumImagery(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path);
bool isCesiumIonImagery(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path);
bool isCesiumPolygonImagery(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path);
bool isCesiumGeoreference(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path);
bool isCesiumIonServer(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path);
bool isCesiumCartographicPolygon(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path);
bool isCesiumSession(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path);
bool hasCesiumGlobeAnchor(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path);
bool isUsdShader(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path);
bool isUsdMaterial(const PXR_NS::UsdStageWeakPtr& pStage, const PXR_NS::SdfPath& path);

struct TranslateRotateScaleOps {
    const PXR_NS::UsdGeomXformOp* pTranslateOp;
    const PXR_NS::UsdGeomXformOp* pRotateOp;
    const PXR_NS::UsdGeomXformOp* pScaleOp;
    MathUtil::EulerAngleOrder eulerAngleOrder;
};

std::optional<TranslateRotateScaleOps> getTranslateRotateScaleOps(const PXR_NS::UsdGeomXformable& xformable);

}; // namespace cesium::omniverse::UsdUtil
