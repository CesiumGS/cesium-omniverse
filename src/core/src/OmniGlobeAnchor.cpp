#include "cesium/omniverse/OmniGlobeAnchor.h"

#include "cesium/omniverse/AssetRegistry.h"
#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/CppUtil.h"
#include "cesium/omniverse/Logger.h"
#include "cesium/omniverse/MathUtil.h"
#include "cesium/omniverse/OmniGeoreference.h"
#include "cesium/omniverse/UsdTokens.h"
#include "cesium/omniverse/UsdUtil.h"

#include <CesiumGeospatial/Cartographic.h>
#include <CesiumGeospatial/Ellipsoid.h>
#include <CesiumGeospatial/GlobeAnchor.h>
#include <CesiumGeospatial/LocalHorizontalCoordinateSystem.h>
#include <CesiumUsdSchemas/data.h>
#include <CesiumUsdSchemas/georeference.h>
#include <CesiumUsdSchemas/globeAnchorAPI.h>
#include <CesiumUsdSchemas/ionServer.h>
#include <CesiumUsdSchemas/rasterOverlay.h>
#include <CesiumUsdSchemas/session.h>
#include <CesiumUsdSchemas/tileset.h>
#include <CesiumUsdSchemas/tokens.h>
#include <CesiumUtility/Math.h>
#include <glm/ext/matrix_relational.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/xformCommonAPI.h>

#include <tuple>
#include <utility>

namespace cesium::omniverse {

OmniGlobeAnchor::OmniGlobeAnchor(Context* pContext, const pxr::SdfPath& path)
    : _pContext(pContext)
    , _path(path) {

    initialize();
}

OmniGlobeAnchor::~OmniGlobeAnchor() = default;

const pxr::SdfPath& OmniGlobeAnchor::getPath() const {
    return _path;
}

bool OmniGlobeAnchor::getDetectTransformChanges() const {
    const auto cesiumGlobeAnchor = UsdUtil::getCesiumGlobeAnchor(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumGlobeAnchor)) {
        return true;
    }

    bool detectTransformChanges;
    cesiumGlobeAnchor.GetDetectTransformChangesAttr().Get(&detectTransformChanges);

    return detectTransformChanges;
}

bool OmniGlobeAnchor::getAdjustOrientation() const {
    const auto cesiumGlobeAnchor = UsdUtil::getCesiumGlobeAnchor(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumGlobeAnchor)) {
        return true;
    }

    bool adjustOrientation;
    cesiumGlobeAnchor.GetAdjustOrientationForGlobeWhenMovingAttr().Get(&adjustOrientation);

    return adjustOrientation;
}

pxr::SdfPath OmniGlobeAnchor::getResolvedGeoreferencePath() const {
    const auto cesiumGlobeAnchor = UsdUtil::getCesiumGlobeAnchor(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumGlobeAnchor)) {
        return {};
    }

    pxr::SdfPathVector targets;
    cesiumGlobeAnchor.GetGeoreferenceBindingRel().GetForwardedTargets(&targets);

    if (!targets.empty()) {
        return targets.front();
    }

    // Fall back to using the first georeference if there's no explicit binding
    const auto pGeoreference = _pContext->getAssetRegistry().getFirstGeoreference();
    if (pGeoreference) {
        return pGeoreference->getPath();
    }

    return {};
}

void OmniGlobeAnchor::updateByEcefPosition() {
    initialize();

    if (!isAnchorValid()) {
        return;
    }

    const auto primLocalToEcefTranslation = getPrimLocalToEcefTranslation();

    const auto tolerance = CesiumUtility::Math::Epsilon4;
    if (MathUtil::epsilonEqual(primLocalToEcefTranslation, _cachedPrimLocalToEcefTranslation, tolerance)) {
        // Short circuit if we don't need to do an actual update.
        return;
    }

    auto primLocalToEcefTransform = _pAnchor->getAnchorToFixedTransform();
    primLocalToEcefTransform[3] = glm::dvec4(primLocalToEcefTranslation, 1.0);

    const auto pGeoreference = _pContext->getAssetRegistry().getGeoreference(getResolvedGeoreferencePath());

    _pAnchor->setAnchorToFixedTransform(
        primLocalToEcefTransform, getAdjustOrientation(), pGeoreference->getEllipsoid());

    finalize();
}

void OmniGlobeAnchor::updateByGeographicCoordinates() {
    initialize();

    if (!isAnchorValid()) {
        return;
    }

    const auto geographicCoordinates = getGeographicCoordinates();

    const auto tolerance = CesiumUtility::Math::Epsilon7;
    if (MathUtil::epsilonEqual(geographicCoordinates, _cachedGeographicCoordinates, tolerance)) {
        // Short circuit if we don't need to do an actual update.
        return;
    }

    const auto pGeoreference = _pContext->getAssetRegistry().getGeoreference(getResolvedGeoreferencePath());
    const auto& ellipsoid = pGeoreference->getEllipsoid();

    const auto primLocalToEcefTranslation = ellipsoid.cartographicToCartesian(geographicCoordinates);

    auto primLocalToEcefTransform = _pAnchor->getAnchorToFixedTransform();
    primLocalToEcefTransform[3] = glm::dvec4(primLocalToEcefTranslation, 1.0);

    _pAnchor->setAnchorToFixedTransform(primLocalToEcefTransform, getAdjustOrientation(), ellipsoid);

    finalize();
}

void OmniGlobeAnchor::updateByPrimLocalTransform(bool resetOrientation) {
    initialize();

    if (!isAnchorValid()) {
        return;
    }

    const auto primLocalTranslation = getPrimLocalTranslation();
    const auto primLocalRotation = getPrimLocalRotation();
    const auto primLocalScale = getPrimLocalScale();

    const auto tolerance = CesiumUtility::Math::Epsilon4;
    if (MathUtil::epsilonEqual(primLocalTranslation, _cachedPrimLocalTranslation, tolerance) &&
        MathUtil::epsilonEqual(primLocalRotation, _cachedPrimLocalRotation, tolerance) &&
        MathUtil::epsilonEqual(primLocalScale, _cachedPrimLocalScale, tolerance)) {
        // Short circuit if we don't need to do an actual update.
        return;
    }

    const auto cesiumGlobeAnchor = UsdUtil::getCesiumGlobeAnchor(_pContext->getUsdStage(), _path);
    const auto xformable = pxr::UsdGeomXformable(cesiumGlobeAnchor);
    const auto xformOps = UsdUtil::getOrCreateTranslateRotateScaleOps(xformable);
    const auto eulerAngleOrder = xformOps->eulerAngleOrder;

    // xform ops are applied right to left, so xformOp:rotateXYZ actually applies a z-axis, then y-axis, then x-axis
    // rotation. To compensate for that, we need to compose euler angles in the reverse order.
    const auto primLocalTransform = MathUtil::composeEuler(
        primLocalTranslation, primLocalRotation, primLocalScale, getReversedEulerAngleOrder(eulerAngleOrder));

    const auto pGeoreference = _pContext->getAssetRegistry().getGeoreference(getResolvedGeoreferencePath());

    _pAnchor->setAnchorToLocalTransform(
        pGeoreference->getLocalCoordinateSystem(),
        primLocalTransform,
        !resetOrientation && getAdjustOrientation(),
        pGeoreference->getEllipsoid());

    finalize();
}

void OmniGlobeAnchor::updateByGeoreference() {
    initialize();

    if (!isAnchorValid()) {
        return;
    }

    finalize();
}

bool OmniGlobeAnchor::isAnchorValid() const {
    const auto georeferencePath = getResolvedGeoreferencePath();

    if (georeferencePath.IsEmpty()) {
        return false;
    }

    const auto pGeoreference = _pContext->getAssetRegistry().getGeoreference(georeferencePath);

    if (!pGeoreference) {
        return false;
    }

    const auto cesiumGlobeAnchor = UsdUtil::getCesiumGlobeAnchor(_pContext->getUsdStage(), _path);
    const auto xformable = pxr::UsdGeomXformable(cesiumGlobeAnchor);
    if (!UsdUtil::isSchemaValid(xformable)) {
        return false;
    }

    const auto xformOps = UsdUtil::getOrCreateTranslateRotateScaleOps(xformable);

    if (!xformOps) {
        _pContext->getLogger()->oneTimeWarning(
            "Globe anchor xform op order must be [translate, rotate, scale] followed by any additional transforms.",
            _path.GetText());
        return false;
    }

    return true;
}

void OmniGlobeAnchor::initialize() {
    if (!isAnchorValid()) {
        _pAnchor = nullptr;
        return;
    }

    if (_pAnchor) {
        return;
    }

    // This function has the effect of baking the world transform into the local transform, which is unavoidable
    // when using globe anchors.

    const auto cesiumGlobeAnchor = UsdUtil::getCesiumGlobeAnchor(_pContext->getUsdStage(), _path);
    const auto xformable = pxr::UsdGeomXformable(cesiumGlobeAnchor);
    if (!UsdUtil::isSchemaValid(xformable)) {
        return;
    }

    bool resetsXformStack;
    const auto originalXformOps = xformable.GetOrderedXformOps(&resetsXformStack);
    const auto translateRotateScaleXformOps = {originalXformOps[0], originalXformOps[1], originalXformOps[2]};

    // Only use translate, rotate, and scale ops when computing the local to ecef transform.
    // Additional transforms like xformOp:rotateX:unitsResolve are not baked into this transform.
    xformable.SetXformOpOrder(translateRotateScaleXformOps);

    // Compute the local to ecef transform
    const auto primLocalToEcefTransform =
        UsdUtil::computePrimLocalToEcefTransform(*_pContext, getResolvedGeoreferencePath(), _path);

    // Now that the transform is computed, switch back to the original ops
    xformable.SetXformOpOrder(originalXformOps);

    // Disable inheriting parent transforms from now on
    xformable.SetResetXformStack(true);

    // Initialize the globe anchor
    _pAnchor = std::make_unique<CesiumGeospatial::GlobeAnchor>(primLocalToEcefTransform);

    // Use the ecef transform (if set) or geographic coordinates (if set) to reposition the globe anchor.
    updateByEcefPosition();
    updateByGeographicCoordinates();

    // Update ecef position, geographic coordinates, and prim local transform from the globe anchor transform
    finalize();
}

void OmniGlobeAnchor::finalize() {
    savePrimLocalToEcefTranslation();
    saveGeographicCoordinates();
    savePrimLocalTransform();
}

glm::dvec3 OmniGlobeAnchor::getPrimLocalToEcefTranslation() const {
    const auto cesiumGlobeAnchor = UsdUtil::getCesiumGlobeAnchor(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumGlobeAnchor)) {
        return {0.0, 0.0, 0.0};
    }

    pxr::GfVec3d primLocalToEcefTranslation;
    cesiumGlobeAnchor.GetPositionAttr().Get(&primLocalToEcefTranslation);

    return UsdUtil::usdToGlmVector(primLocalToEcefTranslation);
}

CesiumGeospatial::Cartographic OmniGlobeAnchor::getGeographicCoordinates() const {
    const auto cesiumGlobeAnchor = UsdUtil::getCesiumGlobeAnchor(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumGlobeAnchor)) {
        return {0.0, 0.0, 0.0};
    }

    double longitude;
    double latitude;
    double height;

    cesiumGlobeAnchor.GetAnchorLongitudeAttr().Get(&longitude);
    cesiumGlobeAnchor.GetAnchorLatitudeAttr().Get(&latitude);
    cesiumGlobeAnchor.GetAnchorHeightAttr().Get(&height);

    return {glm::radians(longitude), glm::radians(latitude), height};
}

glm::dvec3 OmniGlobeAnchor::getPrimLocalTranslation() const {
    const auto cesiumGlobeAnchor = UsdUtil::getCesiumGlobeAnchor(_pContext->getUsdStage(), _path);
    const auto xformable = pxr::UsdGeomXformable(cesiumGlobeAnchor);
    if (!UsdUtil::isSchemaValid(xformable)) {
        return {0.0, 0.0, 0.0};
    }

    const auto xformOps = UsdUtil::getOrCreateTranslateRotateScaleOps(xformable);
    return UsdUtil::getTranslate(xformOps.value().translateOp);
}

glm::dvec3 OmniGlobeAnchor::getPrimLocalRotation() const {
    const auto cesiumGlobeAnchor = UsdUtil::getCesiumGlobeAnchor(_pContext->getUsdStage(), _path);
    const auto xformable = pxr::UsdGeomXformable(cesiumGlobeAnchor);
    if (!UsdUtil::isSchemaValid(xformable)) {
        return {0.0, 0.0, 0.0};
    }

    const auto xformOps = UsdUtil::getOrCreateTranslateRotateScaleOps(xformable);
    return glm::radians(UsdUtil::getRotate(xformOps.value().rotateOp));
}

glm::dvec3 OmniGlobeAnchor::getPrimLocalScale() const {
    const auto cesiumGlobeAnchor = UsdUtil::getCesiumGlobeAnchor(_pContext->getUsdStage(), _path);
    const auto xformable = pxr::UsdGeomXformable(cesiumGlobeAnchor);
    if (!UsdUtil::isSchemaValid(xformable)) {
        return {1.0, 1.0, 1.0};
    }

    const auto xformOps = UsdUtil::getOrCreateTranslateRotateScaleOps(xformable);
    return UsdUtil::getScale(xformOps.value().scaleOp);
}

void OmniGlobeAnchor::savePrimLocalToEcefTranslation() {
    const auto cesiumGlobeAnchor = UsdUtil::getCesiumGlobeAnchor(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumGlobeAnchor)) {
        return;
    }

    const auto& primLocalToEcefTransform = _pAnchor->getAnchorToFixedTransform();
    const auto primLocalToEcefTranslation = glm::dvec3(primLocalToEcefTransform[3]);

    _cachedPrimLocalToEcefTranslation = primLocalToEcefTranslation;

    cesiumGlobeAnchor.GetPositionAttr().Set(UsdUtil::glmToUsdVector(primLocalToEcefTranslation));
}

void OmniGlobeAnchor::saveGeographicCoordinates() {
    const auto cesiumGlobeAnchor = UsdUtil::getCesiumGlobeAnchor(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumGlobeAnchor)) {
        return;
    }

    const auto pGeoreference = _pContext->getAssetRegistry().getGeoreference(getResolvedGeoreferencePath());
    const auto& primLocalToEcefTransform = _pAnchor->getAnchorToFixedTransform();
    const auto primLocalToEcefTranslation = glm::dvec3(primLocalToEcefTransform[3]);
    const auto cartographic = pGeoreference->getEllipsoid().cartesianToCartographic(primLocalToEcefTranslation);

    if (!cartographic) {
        return;
    }

    _cachedGeographicCoordinates = *cartographic;

    cesiumGlobeAnchor.GetAnchorLongitudeAttr().Set(glm::degrees(cartographic->longitude));
    cesiumGlobeAnchor.GetAnchorLatitudeAttr().Set(glm::degrees(cartographic->latitude));
    cesiumGlobeAnchor.GetAnchorHeightAttr().Set(cartographic->height);
}

void OmniGlobeAnchor::savePrimLocalTransform() {
    // Ideally we would just use UsdGeomXformableAPI to set translation, rotation, scale, but this doesn't
    // work when rotation and scale properties are double precision, which is common in Omniverse.

    const auto cesiumGlobeAnchor = UsdUtil::getCesiumGlobeAnchor(_pContext->getUsdStage(), _path);
    const auto xformable = pxr::UsdGeomXformable(cesiumGlobeAnchor);
    if (!UsdUtil::isSchemaValid(xformable)) {
        return;
    }

    auto xformOps = UsdUtil::getOrCreateTranslateRotateScaleOps(xformable);

    auto& [translateOp, rotateOp, scaleOp, eulerAngleOrder] = xformOps.value();

    const auto pGeoreference = _pContext->getAssetRegistry().getGeoreference(getResolvedGeoreferencePath());

    const auto primLocalToWorldTransform =
        _pAnchor->getAnchorToLocalTransform(pGeoreference->getLocalCoordinateSystem());

    // xform ops are applied right to left, so xformOp:rotateXYZ actually applies a z-axis, then y-axis, then x-axis
    // rotation. To compensate for that, we need to decompose euler angles in the reverse order.
    const auto decomposed =
        MathUtil::decomposeEuler(primLocalToWorldTransform, getReversedEulerAngleOrder(eulerAngleOrder));

    _cachedPrimLocalTranslation = decomposed.translation;
    _cachedPrimLocalRotation = decomposed.rotation;
    _cachedPrimLocalScale = decomposed.scale;

    UsdUtil::setTranslate(translateOp, decomposed.translation);
    UsdUtil::setRotate(rotateOp, glm::degrees(decomposed.rotation));
    UsdUtil::setScale(scaleOp, decomposed.scale);
}

} // namespace cesium::omniverse
