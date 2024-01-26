#pragma once

#include "CesiumUsdSchemas/rasterOverlay.h"
#include <glm/fwd.hpp>
#include <pxr/base/gf/declare.h>
#include <pxr/usd/usd/common.h>

PXR_NAMESPACE_OPEN_SCOPE
class CesiumCartographicPolygon;
class CesiumData;
class CesiumGeoreference;
class CesiumGlobeAnchorAPI;
class CesiumRasterOverlay;
class CesiumIonRasterOverlay;
class CesiumIonServer;
class CesiumPolygonRasterOverlay;
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

glm::dvec3 usdToGlmVector(const pxr::GfVec3d& vector);
glm::fvec3 usdToGlmVector(const pxr::GfVec3f& vector);
glm::dmat4 usdToGlmMatrix(const pxr::GfMatrix4d& matrix);
std::array<glm::dvec3, 2> usdToGlmExtent(const pxr::GfRange3d& extent);

pxr::GfVec3d glmToUsdVector(const glm::dvec3& vector);
pxr::GfVec2f glmToUsdVector(const glm::fvec2& vector);
pxr::GfVec3f glmToUsdVector(const glm::fvec3& vector);
pxr::GfVec4f glmToUsdVector(const glm::fvec4& vector);
pxr::GfRange3d glmToUsdExtent(const std::array<glm::dvec3, 2>& extent);
pxr::GfQuatd glmToUsdQuat(const glm::dquat& quat);
pxr::GfQuatf glmToUsdQuat(const glm::fquat& quat);
pxr::GfMatrix4d glmToUsdMatrix(const glm::dmat4& matrix);

glm::dmat4 computePrimLocalToWorldTransform(const pxr::UsdStageWeakPtr& pStage, const pxr::SdfPath& path);
glm::dmat4 computePrimWorldToLocalTransform(const pxr::UsdStageWeakPtr& pStage, const pxr::SdfPath& path);
glm::dmat4 computeEcefToStageTransform(const Context& context, const pxr::SdfPath& georeferencePath);
glm::dmat4 computeEcefToPrimWorldTransform(
    const Context& context,
    const pxr::SdfPath& georeferencePath,
    const pxr::SdfPath& primPath);
glm::dmat4 computePrimWorldToEcefTransform(
    const Context& context,
    const pxr::SdfPath& georeferencePath,
    const pxr::SdfPath& primPath);
glm::dmat4 computeEcefToPrimLocalTransform(
    const Context& context,
    const pxr::SdfPath& georeferencePath,
    const pxr::SdfPath& primPath);
glm::dmat4 computePrimLocalToEcefTransform(
    const Context& context,
    const pxr::SdfPath& georeferencePath,
    const pxr::SdfPath& primPath);
CesiumGeospatial::LocalHorizontalCoordinateSystem computeLocalCoordinateSystem(
    const pxr::UsdStageWeakPtr& pStage,
    const CesiumGeospatial::Cartographic& origin,
    const CesiumGeospatial::Ellipsoid& ellipsoid);

Cesium3DTilesSelection::ViewState computeViewState(
    const Context& context,
    const pxr::SdfPath& georeferencePath,
    const pxr::SdfPath& primPath,
    const Viewport& viewport);

bool primExists(const pxr::UsdStageWeakPtr& pStage, const pxr::SdfPath& path);
bool isPrimVisible(const pxr::UsdStageWeakPtr& pStage, const pxr::SdfPath& path);
const std::string& getName(const pxr::UsdStageWeakPtr& pStage, const pxr::SdfPath& path);

pxr::TfToken getUsdUpAxis(const pxr::UsdStageWeakPtr& pStage);
double getUsdMetersPerUnit(const pxr::UsdStageWeakPtr& pStage);
pxr::SdfPath getRootPath(const pxr::UsdStageWeakPtr& pStage);
pxr::SdfPath
makeUniquePath(const pxr::UsdStageWeakPtr& pStage, const pxr::SdfPath& parentPath, const std::string& name);
std::string getSafeName(const std::string& name);
pxr::TfToken getDynamicTextureProviderAssetPathToken(const std::string_view& name);

pxr::CesiumData defineCesiumData(const pxr::UsdStageWeakPtr& pStage, const pxr::SdfPath& path);
pxr::CesiumTileset defineCesiumTileset(const pxr::UsdStageWeakPtr& pStage, const pxr::SdfPath& path);
pxr::CesiumIonRasterOverlay defineCesiumIonImagery(const pxr::UsdStageWeakPtr& pStage, const pxr::SdfPath& path);
pxr::CesiumPolygonRasterOverlay defineCesiumPolygonImagery(const pxr::UsdStageWeakPtr& pStage, const pxr::SdfPath& path);
pxr::CesiumGeoreference defineCesiumGeoreference(const pxr::UsdStageWeakPtr& pStage, const pxr::SdfPath& path);
pxr::CesiumIonServer defineCesiumIonServer(const pxr::UsdStageWeakPtr& pStage, const pxr::SdfPath& path);
pxr::CesiumGlobeAnchorAPI applyCesiumGlobeAnchor(const pxr::UsdStageWeakPtr& pStage, const pxr::SdfPath& path);
pxr::CesiumSession defineCesiumSession(const pxr::UsdStageWeakPtr& pStage, const pxr::SdfPath& path);

pxr::CesiumData getCesiumData(const pxr::UsdStageWeakPtr& pStage, const pxr::SdfPath& path);
pxr::CesiumTileset getCesiumTileset(const pxr::UsdStageWeakPtr& pStage, const pxr::SdfPath& path);
pxr::CesiumRasterOverlay getCesiumRasterOverlay(const pxr::UsdStageWeakPtr& pStage, const pxr::SdfPath& path);
pxr::CesiumIonRasterOverlay getCesiumIonRasterOverlay(const pxr::UsdStageWeakPtr& pStage, const pxr::SdfPath& path);
pxr::CesiumPolygonRasterOverlay getCesiumPolygonRasterOverlay(const pxr::UsdStageWeakPtr& pStage, const pxr::SdfPath& path);
pxr::CesiumGeoreference getCesiumGeoreference(const pxr::UsdStageWeakPtr& pStage, const pxr::SdfPath& path);
pxr::CesiumGlobeAnchorAPI getCesiumGlobeAnchor(const pxr::UsdStageWeakPtr& pStage, const pxr::SdfPath& path);
pxr::CesiumIonServer getCesiumIonServer(const pxr::UsdStageWeakPtr& pStage, const pxr::SdfPath& path);
pxr::CesiumCartographicPolygon
getCesiumCartographicPolygon(const pxr::UsdStageWeakPtr& pStage, const pxr::SdfPath& path);
pxr::UsdShadeShader getUsdShader(const pxr::UsdStageWeakPtr& pStage, const pxr::SdfPath& path);
pxr::UsdGeomBasisCurves getUsdBasisCurves(const pxr::UsdStageWeakPtr& pStage, const pxr::SdfPath& path);

pxr::CesiumSession getOrCreateCesiumSession(const pxr::UsdStageWeakPtr& pStage);

bool isCesiumData(const pxr::UsdStageWeakPtr& pStage, const pxr::SdfPath& path);
bool isCesiumTileset(const pxr::UsdStageWeakPtr& pStage, const pxr::SdfPath& path);
bool isCesiumRasterOverlay(const pxr::UsdStageWeakPtr& pStage, const pxr::SdfPath& path);
bool isCesiumIonRasterOverlay(const pxr::UsdStageWeakPtr& pStage, const pxr::SdfPath& path);
bool isCesiumPolygonRasterOverlay(const pxr::UsdStageWeakPtr& pStage, const pxr::SdfPath& path);
bool isCesiumGeoreference(const pxr::UsdStageWeakPtr& pStage, const pxr::SdfPath& path);
bool isCesiumIonServer(const pxr::UsdStageWeakPtr& pStage, const pxr::SdfPath& path);
bool isCesiumCartographicPolygon(const pxr::UsdStageWeakPtr& pStage, const pxr::SdfPath& path);
bool isCesiumSession(const pxr::UsdStageWeakPtr& pStage, const pxr::SdfPath& path);
bool hasCesiumGlobeAnchor(const pxr::UsdStageWeakPtr& pStage, const pxr::SdfPath& path);
bool isUsdShader(const pxr::UsdStageWeakPtr& pStage, const pxr::SdfPath& path);
bool isUsdMaterial(const pxr::UsdStageWeakPtr& pStage, const pxr::SdfPath& path);

struct TranslateRotateScaleOps {
    const pxr::UsdGeomXformOp* pTranslateOp;
    const pxr::UsdGeomXformOp* pRotateOp;
    const pxr::UsdGeomXformOp* pScaleOp;
    MathUtil::EulerAngleOrder eulerAngleOrder;
};

std::optional<TranslateRotateScaleOps> getTranslateRotateScaleOps(const pxr::UsdGeomXformable& xformable);

}; // namespace cesium::omniverse::UsdUtil
