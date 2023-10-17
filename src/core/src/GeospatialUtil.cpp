#include "cesium/omniverse/GeospatialUtil.h"

#include "cesium/omniverse/GlobeAnchorRegistry.h"
#include "cesium/omniverse/LoggerSink.h"
#include "cesium/omniverse/OmniGlobeAnchor.h"
#include "cesium/omniverse/UsdUtil.h"

#include <CesiumGeospatial/GlobeAnchor.h>
#include <pxr/usd/usdGeom/xform.h>

namespace cesium::omniverse::GeospatialUtil {

CesiumGeospatial::Cartographic convertGeoreferenceToCartographic(const pxr::CesiumGeoreference& georeference) {
    double longitude;
    double latitude;
    double height;
    georeference.GetGeoreferenceOriginLongitudeAttr().Get<double>(&longitude);
    georeference.GetGeoreferenceOriginLatitudeAttr().Get<double>(&latitude);
    georeference.GetGeoreferenceOriginHeightAttr().Get<double>(&height);

    return {glm::radians(longitude), glm::radians(latitude), height};
}

CesiumGeospatial::LocalHorizontalCoordinateSystem getCoordinateSystem(const CesiumGeospatial::Cartographic& origin) {
    const auto upAxis = UsdUtil::getUsdUpAxis();
    const auto scaleInMeters = UsdUtil::getUsdMetersPerUnit();

    if (upAxis == pxr::UsdGeomTokens->z) {
        return {
            origin,
            CesiumGeospatial::LocalDirection::East,
            CesiumGeospatial::LocalDirection::North,
            CesiumGeospatial::LocalDirection::Up,
            scaleInMeters};
    }

    return {
        origin,
        CesiumGeospatial::LocalDirection::East,
        CesiumGeospatial::LocalDirection::Up,
        CesiumGeospatial::LocalDirection::South,
        scaleInMeters};
}

void updateAnchorByUsdTransform(
    const CesiumGeospatial::Cartographic& origin,
    const pxr::CesiumGlobeAnchorAPI& anchorApi) {
    std::optional<std::shared_ptr<OmniGlobeAnchor>> maybeGlobeAnchor =
        GlobeAnchorRegistry::getInstance().getAnchor(anchorApi.GetPath());

    std::shared_ptr<OmniGlobeAnchor> globeAnchor;
    if (maybeGlobeAnchor.has_value()) {
        globeAnchor = maybeGlobeAnchor.value();

        auto usdTransform = UsdUtil::getCesiumTransformOpValueForPathIfExists(globeAnchor->getPrimPath());
        auto cachedTransform = globeAnchor->getCachedTransformation();

        double tolerance = 0.01;
        if (usdTransform.has_value() && pxr::GfIsClose(usdTransform.value(), cachedTransform, tolerance)) {

            // Short circuit if an update isn't actually necessary.
            return;
        }

        bool shouldReorient;
        anchorApi.GetAdjustOrientationForGlobeWhenMovingAttr().Get(&shouldReorient);
        globeAnchor->updateByUsdTransform(origin, shouldReorient);
    } else {
        auto anchorToFixed = UsdUtil::computeUsdLocalToEcefTransformForPrim(origin, anchorApi.GetPath());
        globeAnchor = GlobeAnchorRegistry::getInstance().createAnchor(anchorApi.GetPath(), anchorToFixed);
    }

    auto fixedTransform = UsdUtil::glmToUsdMatrixDecomposed(globeAnchor->getAnchorToFixedTransform());

    if (!maybeGlobeAnchor.has_value()) {
        auto localTransform = globeAnchor->getAnchorToLocalTransform(origin);
        UsdUtil::addOrUpdateTransformOpForAnchor(anchorApi.GetPath(), localTransform);
    }

    anchorApi.GetPositionAttr().Set(fixedTransform.position);
    anchorApi.GetRotationAttr().Set(pxr::GfVec3d(UsdUtil::getEulerAnglesFromQuaternion(fixedTransform.orientation)));
    anchorApi.GetScaleAttr().Set(pxr::GfVec3d(fixedTransform.scale));

    std::optional<CesiumGeospatial::Cartographic> cartographicPosition =
        CesiumGeospatial::Ellipsoid::WGS84.cartesianToCartographic(UsdUtil::usdToGlmVector(fixedTransform.position));

    if (cartographicPosition) {
        auto geographicPosition = pxr::GfVec3d{
            glm::degrees(cartographicPosition->latitude),
            glm::degrees(cartographicPosition->longitude),
            cartographicPosition->height};

        anchorApi.GetGeographicCoordinateAttr().Set(geographicPosition);
    } else {
        anchorApi.GetGeographicCoordinateAttr().Set(pxr::GfVec3d{0.0, 0.0, 0.0});
        CESIUM_LOG_WARN("Invalid cartographic position for Anchor. Reset to 0, 0, 0.");
    }

    globeAnchor->updateCachedValues();
}

void updateAnchorByLatLongHeight(
    const CesiumGeospatial::Cartographic& origin,
    const pxr::CesiumGlobeAnchorAPI& anchorApi) {
    std::optional<std::shared_ptr<OmniGlobeAnchor>> maybeGlobeAnchor =
        GlobeAnchorRegistry::getInstance().getAnchor(anchorApi.GetPath());

    if (!maybeGlobeAnchor.has_value()) {
        CESIUM_LOG_ERROR(
            "Anchor does not exist in registry but exists in stage. Path: {}", anchorApi.GetPath().GetString());

        return;
    }

    std::shared_ptr<OmniGlobeAnchor> globeAnchor = maybeGlobeAnchor.value();

    pxr::GfVec3d usdGeographicCoordinate;
    anchorApi.GetGeographicCoordinateAttr().Get(&usdGeographicCoordinate);

    auto cachedGeographicCoordinate = globeAnchor->getCachedGeographicCoordinate();

    double tolerance = 0.0000001;
    if (pxr::GfIsClose(usdGeographicCoordinate, cachedGeographicCoordinate, tolerance)) {

        // Short circuit if we don't need to do an actual update.
        return;
    }

    double usdLatitude = usdGeographicCoordinate[0];
    double usdLongitude = usdGeographicCoordinate[1];
    double usdHeight = usdGeographicCoordinate[2];

    bool shouldReorient;
    anchorApi.GetAdjustOrientationForGlobeWhenMovingAttr().Get(&shouldReorient);

    auto cartographic = CesiumGeospatial::Cartographic::fromDegrees(usdLongitude, usdLatitude, usdHeight);
    globeAnchor->updateByGeographicCoordinates(cartographic, shouldReorient);

    auto localTransform = globeAnchor->getAnchorToLocalTransform(origin);
    UsdUtil::addOrUpdateTransformOpForAnchor(anchorApi.GetPath(), localTransform);

    auto fixedTransform = UsdUtil::glmToUsdMatrixDecomposed(globeAnchor->getAnchorToFixedTransform());
    anchorApi.GetPositionAttr().Set(fixedTransform.position);
    anchorApi.GetRotationAttr().Set(pxr::GfVec3d(UsdUtil::getEulerAnglesFromQuaternion(fixedTransform.orientation)));
    anchorApi.GetScaleAttr().Set(pxr::GfVec3d(fixedTransform.scale));

    globeAnchor->updateCachedValues();
}

void updateAnchorByFixedTransform(
    const CesiumGeospatial::Cartographic& origin,
    const pxr::CesiumGlobeAnchorAPI& anchorApi) {
    std::optional<std::shared_ptr<OmniGlobeAnchor>> maybeGlobeAnchor =
        GlobeAnchorRegistry::getInstance().getAnchor(anchorApi.GetPath());

    if (!maybeGlobeAnchor.has_value()) {
        CESIUM_LOG_ERROR(
            "Anchor does not exist in registry but exists in stage. Path: {}", anchorApi.GetPath().GetString());

        return;
    }

    std::shared_ptr<OmniGlobeAnchor> globeAnchor = maybeGlobeAnchor.value();

    pxr::GfVec3d usdEcefPositionVec;
    anchorApi.GetPositionAttr().Get(&usdEcefPositionVec);
    auto ecefPositionVec = UsdUtil::usdToGlmVector(usdEcefPositionVec);

    pxr::GfVec3d usdEcefRotationVec;
    anchorApi.GetRotationAttr().Get(&usdEcefRotationVec);
    auto ecefRotationVec = UsdUtil::usdToGlmVector(usdEcefRotationVec);

    pxr::GfVec3d usdEcefScaleVec;
    anchorApi.GetScaleAttr().Get(&usdEcefScaleVec);
    auto ecefScaleVec = UsdUtil::usdToGlmVector(usdEcefScaleVec);

    auto cachedEcefPosition = globeAnchor->getCachedEcefPosition();
    auto cachedEcefRotation = globeAnchor->getCachedEcefRotation();
    auto cachedEcefScale = globeAnchor->getCachedEcefScale();

    double tolerance = 0.0001;
    if (pxr::GfIsClose(usdEcefPositionVec, cachedEcefPosition, tolerance) &&
        pxr::GfIsClose(usdEcefRotationVec, cachedEcefRotation, tolerance) &&
        pxr::GfIsClose(usdEcefScaleVec, cachedEcefScale, tolerance)) {

        // Short circuit early if there isn't actually an update.
        return;
    }

    bool shouldReorient;
    anchorApi.GetAdjustOrientationForGlobeWhenMovingAttr().Get(&shouldReorient);
    globeAnchor->updateByFixedTransform(ecefPositionVec, glm::radians(ecefRotationVec), ecefScaleVec, shouldReorient);

    auto localTransform = globeAnchor->getAnchorToLocalTransform(origin);
    UsdUtil::addOrUpdateTransformOpForAnchor(anchorApi.GetPath(), localTransform);

    auto cartographicPosition = globeAnchor->getCartographicPosition();

    if (cartographicPosition) {
        auto geographicPosition = pxr::GfVec3d{
            glm::degrees(cartographicPosition->latitude),
            glm::degrees(cartographicPosition->longitude),
            cartographicPosition->height};

        anchorApi.GetGeographicCoordinateAttr().Set(geographicPosition);
    } else {
        anchorApi.GetGeographicCoordinateAttr().Set(pxr::GfVec3d{0.0, 0.0, 0.0});
        CESIUM_LOG_WARN("Invalid cartographic position for Anchor. Reset to 0, 0, 0.");
    }

    globeAnchor->updateCachedValues();
}

void updateAnchorOrigin(
    const CesiumGeospatial::Cartographic& origin,
    const pxr::CesiumGlobeAnchorAPI& anchorApi,
    const std::shared_ptr<OmniGlobeAnchor>& globeAnchor) {
    auto localTransform = globeAnchor->getAnchorToLocalTransform(origin);
    UsdUtil::addOrUpdateTransformOpForAnchor(anchorApi.GetPath(), localTransform);

    globeAnchor->updateCachedValues();
}

} // namespace cesium::omniverse::GeospatialUtil
