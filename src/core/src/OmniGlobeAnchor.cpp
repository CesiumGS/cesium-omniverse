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
#include <CesiumUsdSchemas/imagery.h>
#include <CesiumUsdSchemas/ionServer.h>
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

OmniGlobeAnchor::OmniGlobeAnchor(Context* pContext, const PXR_NS::SdfPath& path)
    : _pContext(pContext)
    , _path(path) {

    // Use the geographic coordinates (if set) or ecef transform (if set) to initialize the globe anchor.
    // If neither are set, initialize the globe anchor from the prim's local transform.
    updateByGeographicCoordinates(false);
    updateByPrimLocalToEcefTransform(false);
    updateByPrimLocalTransform(true);
}

OmniGlobeAnchor::~OmniGlobeAnchor() = default;

bool OmniGlobeAnchor::isGeoreferenceValid() const {
    const auto georeferencePath = getGeoreferencePath();

    if (georeferencePath.IsEmpty()) {
        return false;
    }

    const auto pGeoreference = _pContext->getAssetRegistry().getGeoreference(georeferencePath);

    if (!pGeoreference) {
        return false;
    }

    return true;
}

bool OmniGlobeAnchor::initialize() {
    if (!isGeoreferenceValid()) {
        _pAnchor = nullptr;
        return false;
    }

    const auto cesiumGlobeAnchor = UsdUtil::getCesiumGlobeAnchor(_pContext->getUsdStage(), _path);
    const auto xformable = PXR_NS::UsdGeomXformable(cesiumGlobeAnchor.GetPrim());
    const auto xformOps = UsdUtil::getTranslateRotateScaleOps(xformable);

    if (!xformOps) {
        _pContext->getLogger()->oneTimeWarning(fmt::format(
            "Globe anchor xform op order must [translate, rotate, scale] without additional transforms.",
            _path.GetText()));
        return false;
    }

    if (!_pAnchor) {
        const auto anchorToFixed = UsdUtil::computePrimLocalToEcefTransform(*_pContext, getGeoreferencePath(), _path);
        _pAnchor = std::make_unique<CesiumGeospatial::GlobeAnchor>(anchorToFixed);
    }

    return true;
}

const PXR_NS::SdfPath& OmniGlobeAnchor::getPath() const {
    return _path;
}

bool OmniGlobeAnchor::getDetectTransformChanges() const {
    const auto cesiumGlobeAnchor = UsdUtil::getCesiumGlobeAnchor(_pContext->getUsdStage(), _path);

    bool detectTransformChanges;
    cesiumGlobeAnchor.GetDetectTransformChangesAttr().Get(&detectTransformChanges);

    return detectTransformChanges;
}

PXR_NS::SdfPath OmniGlobeAnchor::getGeoreferencePath() const {
    const auto cesiumGlobeAnchor = UsdUtil::getCesiumGlobeAnchor(_pContext->getUsdStage(), _path);

    PXR_NS::SdfPathVector targets;
    cesiumGlobeAnchor.GetGeoreferenceBindingRel().GetForwardedTargets(&targets);

    if (targets.empty()) {
        return {};
    }

    return targets.front();
}

void OmniGlobeAnchor::updateByGeographicCoordinates(bool force) {
    if (!initialize()) {
        return;
    }

    const auto geographicCoordinates = getGeographicCoordinates();

    const auto tolerance = CesiumUtility::Math::Epsilon7;
    if (!force && MathUtil::epsilonEqual(geographicCoordinates, _cachedGeographicCoordinates, tolerance)) {
        // Short circuit if we don't need to do an actual update.
        return;
    }

    const auto pGeoreference = _pContext->getAssetRegistry().getGeoreference(getGeoreferencePath());
    const auto& ellipsoid = pGeoreference->getEllipsoid();

    const auto primLocalToEcefTranslation = ellipsoid.cartographicToCartesian(geographicCoordinates);
    const auto primLocalToEcefRotation = getPrimLocalToEcefRotation();
    const auto primLocalToEcefScale = getPrimLocalToEcefScale();

    const auto primLocalToEcefTransform = MathUtil::composeEuler(
        primLocalToEcefTranslation, primLocalToEcefRotation, primLocalToEcefScale, MathUtil::EulerAngleOrder::XYZ);

    _pAnchor->setAnchorToFixedTransform(primLocalToEcefTransform, getAdjustOrientation(), ellipsoid);

    finalize();
}

void OmniGlobeAnchor::updateByPrimLocalTransform(bool force) {
    if (!initialize()) {
        return;
    }

    const auto cesiumGlobeAnchor = UsdUtil::getCesiumGlobeAnchor(_pContext->getUsdStage(), _path);
    const auto xformable = PXR_NS::UsdGeomXformable(cesiumGlobeAnchor.GetPrim());
    const auto xformOps = UsdUtil::getTranslateRotateScaleOps(xformable);
    const auto eulerAngleOrder = xformOps->eulerAngleOrder;

    const auto primLocalTranslation = getPrimLocalTranslation();
    const auto primLocalRotation = getPrimLocalRotation();
    const auto primLocalScale = getPrimLocalScale();
    const auto primLocalTransform =
        MathUtil::composeEuler(primLocalTranslation, primLocalRotation, primLocalScale, eulerAngleOrder);

    const auto tolerance = CesiumUtility::Math::Epsilon4;
    if (!force && MathUtil::epsilonEqual(primLocalTranslation, _cachedPrimLocalTranslation, tolerance) &&
        MathUtil::epsilonEqual(primLocalRotation, _cachedPrimLocalRotation, tolerance) &&
        MathUtil::epsilonEqual(primLocalScale, _cachedPrimLocalScale, tolerance)) {
        // Short circuit if we don't need to do an actual update.
        return;
    }

    const auto pGeoreference = _pContext->getAssetRegistry().getGeoreference(getGeoreferencePath());

    _pAnchor->setAnchorToLocalTransform(
        pGeoreference->getLocalCoordinateSystem(),
        primLocalTransform,
        getAdjustOrientation(),
        pGeoreference->getEllipsoid());

    finalize();
}

void OmniGlobeAnchor::updateByPrimLocalToEcefTransform(bool force) {
    if (!initialize()) {
        return;
    }

    const auto primLocalToEcefTranslation = getPrimLocalToEcefTranslation();
    const auto primLocalToEcefRotation = getPrimLocalToEcefRotation();
    const auto primLocalToEcefScale = getPrimLocalToEcefScale();

    const auto tolerance = CesiumUtility::Math::Epsilon4;
    if (!force && MathUtil::epsilonEqual(primLocalToEcefTranslation, _cachedPrimLocalToEcefTranslation, tolerance) &&
        MathUtil::epsilonEqual(primLocalToEcefRotation, _cachedPrimLocalToEcefRotation, tolerance) &&
        MathUtil::epsilonEqual(primLocalToEcefScale, _cachedPrimLocalToEcefScale, tolerance)) {
        // Short circuit if we don't need to do an actual update.
        return;
    }

    const auto primLocalToEcefTransform = MathUtil::composeEuler(
        primLocalToEcefTranslation, primLocalToEcefRotation, primLocalToEcefScale, MathUtil::EulerAngleOrder::XYZ);

    const auto pGeoreference = _pContext->getAssetRegistry().getGeoreference(getGeoreferencePath());

    _pAnchor->setAnchorToFixedTransform(
        primLocalToEcefTransform, getAdjustOrientation(), pGeoreference->getEllipsoid());

    finalize();
}

void OmniGlobeAnchor::updateByGeoreference() {
    if (!initialize()) {
        return;
    }

    const auto primLocalToEcefTranslation = getPrimLocalToEcefTranslation();
    const auto primLocalToEcefRotation = getPrimLocalToEcefRotation();
    const auto primLocalToEcefScale = getPrimLocalToEcefScale();

    const auto primLocalToEcefTransform = MathUtil::composeEuler(
        primLocalToEcefTranslation, primLocalToEcefRotation, primLocalToEcefScale, MathUtil::EulerAngleOrder::XYZ);

    const auto pGeoreference = _pContext->getAssetRegistry().getGeoreference(getGeoreferencePath());

    _pAnchor->setAnchorToFixedTransform(
        primLocalToEcefTransform, getAdjustOrientation(), pGeoreference->getEllipsoid());

    finalize();
}

void OmniGlobeAnchor::finalize() {
    saveGeographicCoordinates();
    savePrimLocalToEcefTransform();
    savePrimLocalTransform();
}

bool OmniGlobeAnchor::getAdjustOrientation() const {
    const auto cesiumGlobeAnchor = UsdUtil::getCesiumGlobeAnchor(_pContext->getUsdStage(), _path);

    bool adjustOrientation;
    cesiumGlobeAnchor.GetAdjustOrientationForGlobeWhenMovingAttr().Get(&adjustOrientation);

    return adjustOrientation;
}

CesiumGeospatial::Cartographic OmniGlobeAnchor::getGeographicCoordinates() const {
    const auto cesiumGlobeAnchor = UsdUtil::getCesiumGlobeAnchor(_pContext->getUsdStage(), _path);

    PXR_NS::GfVec3d coordinates;
    cesiumGlobeAnchor.GetGeographicCoordinateAttr().Get(&coordinates);

    const auto longitude = glm::radians(coordinates[1]);
    const auto latitude = glm::radians(coordinates[0]);
    const auto height = coordinates[2];

    return {longitude, latitude, height};
}

glm::dvec3 OmniGlobeAnchor::getPrimLocalTranslation() const {
    const auto cesiumGlobeAnchor = UsdUtil::getCesiumGlobeAnchor(_pContext->getUsdStage(), _path);
    const auto xformable = PXR_NS::UsdGeomXformable(cesiumGlobeAnchor.GetPrim());
    const auto xformOps = UsdUtil::getTranslateRotateScaleOps(xformable);
    const auto pTranslateOp = xformOps.value().pTranslateOp;

    if (pTranslateOp->GetPrecision() == PXR_NS::UsdGeomXformOp::PrecisionDouble) {
        PXR_NS::GfVec3d translation;
        pTranslateOp->Get(&translation);
        return UsdUtil::usdToGlmVector(translation);
    } else if (pTranslateOp->GetPrecision() == PXR_NS::UsdGeomXformOp::PrecisionFloat) {
        PXR_NS::GfVec3f translation;
        pTranslateOp->Get(&translation);
        return glm::dvec3(UsdUtil::usdToGlmVector(translation));
    }

    return glm::dvec3(0.0);
}

glm::dvec3 OmniGlobeAnchor::getPrimLocalRotation() const {
    const auto cesiumGlobeAnchor = UsdUtil::getCesiumGlobeAnchor(_pContext->getUsdStage(), _path);
    const auto xformable = PXR_NS::UsdGeomXformable(cesiumGlobeAnchor.GetPrim());
    const auto xformOps = UsdUtil::getTranslateRotateScaleOps(xformable);
    const auto pRotateOp = xformOps.value().pRotateOp;

    if (pRotateOp->GetPrecision() == PXR_NS::UsdGeomXformOp::PrecisionDouble) {
        PXR_NS::GfVec3d rotation;
        pRotateOp->Get(&rotation);
        return glm::radians(UsdUtil::usdToGlmVector(rotation));
    } else if (pRotateOp->GetPrecision() == PXR_NS::UsdGeomXformOp::PrecisionFloat) {
        PXR_NS::GfVec3f rotation;
        pRotateOp->Get(&rotation);
        return glm::radians(glm::dvec3(UsdUtil::usdToGlmVector(rotation)));
    }

    return glm::dvec3(0.0);
}

glm::dvec3 OmniGlobeAnchor::getPrimLocalScale() const {
    const auto cesiumGlobeAnchor = UsdUtil::getCesiumGlobeAnchor(_pContext->getUsdStage(), _path);
    const auto xformable = PXR_NS::UsdGeomXformable(cesiumGlobeAnchor.GetPrim());
    const auto xformOps = UsdUtil::getTranslateRotateScaleOps(xformable);
    const auto pScaleOp = xformOps.value().pScaleOp;

    if (pScaleOp->GetPrecision() == PXR_NS::UsdGeomXformOp::PrecisionDouble) {
        PXR_NS::GfVec3d scale;
        pScaleOp->Get(&scale);
        return UsdUtil::usdToGlmVector(scale);
    } else if (pScaleOp->GetPrecision() == PXR_NS::UsdGeomXformOp::PrecisionFloat) {
        PXR_NS::GfVec3f scale;
        pScaleOp->Get(&scale);
        return glm::dvec3(UsdUtil::usdToGlmVector(scale));
    }

    return glm::dvec3(1.0);
}

glm::dvec3 OmniGlobeAnchor::getPrimLocalToEcefTranslation() const {
    const auto cesiumGlobeAnchor = UsdUtil::getCesiumGlobeAnchor(_pContext->getUsdStage(), _path);

    PXR_NS::GfVec3d primLocalToEcefTranslation;
    cesiumGlobeAnchor.GetPositionAttr().Get(&primLocalToEcefTranslation);

    return UsdUtil::usdToGlmVector(primLocalToEcefTranslation);
}

glm::dvec3 OmniGlobeAnchor::getPrimLocalToEcefRotation() const {
    const auto cesiumGlobeAnchor = UsdUtil::getCesiumGlobeAnchor(_pContext->getUsdStage(), _path);

    PXR_NS::GfVec3d primLocalToEcefRotation;
    cesiumGlobeAnchor.GetRotationAttr().Get(&primLocalToEcefRotation);

    return UsdUtil::usdToGlmVector(primLocalToEcefRotation);
}

glm::dvec3 OmniGlobeAnchor::getPrimLocalToEcefScale() const {
    const auto cesiumGlobeAnchor = UsdUtil::getCesiumGlobeAnchor(_pContext->getUsdStage(), _path);

    PXR_NS::GfVec3d primLocalToEcefScale;
    cesiumGlobeAnchor.GetScaleAttr().Get(&primLocalToEcefScale);

    return UsdUtil::usdToGlmVector(primLocalToEcefScale);
}

void OmniGlobeAnchor::saveGeographicCoordinates() {
    const auto pGeoreference = _pContext->getAssetRegistry().getGeoreference(getGeoreferencePath());
    const auto& anchorToFixed = _pAnchor->getAnchorToFixedTransform();
    const auto ecefPosition = glm::dvec3(anchorToFixed[3]);
    const auto cartographic = pGeoreference->getEllipsoid().cartesianToCartographic(ecefPosition);

    if (!cartographic) {
        return;
    }

    _cachedGeographicCoordinates = *cartographic;

    const auto cesiumGlobeAnchor = UsdUtil::getCesiumGlobeAnchor(_pContext->getUsdStage(), _path);
    cesiumGlobeAnchor.GetGeographicCoordinateAttr().Set(PXR_NS::GfVec3d(
        glm::degrees(cartographic->latitude), glm::degrees(cartographic->longitude), cartographic->height));
}

void OmniGlobeAnchor::savePrimLocalToEcefTransform() {
    const auto& primLocalToEcefTransform = _pAnchor->getAnchorToFixedTransform();
    const auto decomposed = MathUtil::decomposeEuler(primLocalToEcefTransform, MathUtil::EulerAngleOrder::XYZ);

    _cachedPrimLocalToEcefTranslation = decomposed.translation;
    _cachedPrimLocalToEcefRotation = decomposed.rotation;
    _cachedPrimLocalToEcefScale = decomposed.scale;

    const auto cesiumGlobeAnchor = UsdUtil::getCesiumGlobeAnchor(_pContext->getUsdStage(), _path);
    cesiumGlobeAnchor.GetPositionAttr().Set(UsdUtil::glmToUsdVector(decomposed.translation));
    cesiumGlobeAnchor.GetRotationAttr().Set(UsdUtil::glmToUsdVector(decomposed.rotation));
    cesiumGlobeAnchor.GetScaleAttr().Set(UsdUtil::glmToUsdVector(decomposed.scale));
}

void OmniGlobeAnchor::savePrimLocalTransform() {
    // Ideally we would just use UsdGeomXformableAPI to set translation, rotation, scale, but this doesn't
    // work when rotation and scale properties are double precision, which is common in Omniverse.

    const auto pGeoreference = _pContext->getAssetRegistry().getGeoreference(getGeoreferencePath());
    const auto anchorToLocalTransform = _pAnchor->getAnchorToLocalTransform(pGeoreference->getLocalCoordinateSystem());
    const auto cesiumGlobeAnchor = UsdUtil::getCesiumGlobeAnchor(_pContext->getUsdStage(), _path);
    const auto xformable = PXR_NS::UsdGeomXformable(cesiumGlobeAnchor.GetPrim());
    const auto xformOps = UsdUtil::getTranslateRotateScaleOps(xformable);

    const auto& [pTranslateOp, pRotateOp, pScaleOp, eulerAngleOrder] = xformOps.value();

    const auto decomposed = MathUtil::decomposeEuler(anchorToLocalTransform, eulerAngleOrder);

    _cachedPrimLocalTranslation = decomposed.translation;
    _cachedPrimLocalRotation = decomposed.rotation;
    _cachedPrimLocalScale = decomposed.scale;

    if (pTranslateOp->GetPrecision() == PXR_NS::UsdGeomXformOp::PrecisionDouble) {
        pTranslateOp->Set(UsdUtil::glmToUsdVector(decomposed.translation));
    } else if (pTranslateOp->GetPrecision() == PXR_NS::UsdGeomXformOp::PrecisionFloat) {
        pTranslateOp->Set(UsdUtil::glmToUsdVector(glm::fvec3(decomposed.translation)));
    }

    if (pRotateOp->GetPrecision() == PXR_NS::UsdGeomXformOp::PrecisionDouble) {
        pRotateOp->Set(UsdUtil::glmToUsdVector(glm::degrees(decomposed.rotation)));
    } else if (pRotateOp->GetPrecision() == PXR_NS::UsdGeomXformOp::PrecisionFloat) {
        pRotateOp->Set(UsdUtil::glmToUsdVector(glm::fvec3(glm::degrees(decomposed.rotation))));
    }

    if (pScaleOp->GetPrecision() == PXR_NS::UsdGeomXformOp::PrecisionDouble) {
        pScaleOp->Set(UsdUtil::glmToUsdVector(decomposed.scale));
    } else if (pScaleOp->GetPrecision() == PXR_NS::UsdGeomXformOp::PrecisionFloat) {
        pScaleOp->Set(UsdUtil::glmToUsdVector(glm::fvec3(decomposed.scale)));
    }
}

} // namespace cesium::omniverse
