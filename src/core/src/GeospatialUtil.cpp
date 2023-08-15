#include "cesium/omniverse/GeospatialUtil.h"

#include "cesium/omniverse/GlobeAnchorRegistry.h"
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

[[maybe_unused]] CesiumGeospatial::LocalHorizontalCoordinateSystem
getCoordinateSystem(const pxr::CesiumGeoreference& georeference, const double scaleInMeters) {
    auto origin = GeospatialUtil::convertGeoreferenceToCartographic(georeference);
    return getCoordinateSystem(origin, scaleInMeters);
}

CesiumGeospatial::LocalHorizontalCoordinateSystem
getCoordinateSystem(const CesiumGeospatial::Cartographic& origin, const double scaleInMeters) {
    const auto upAxis = UsdUtil::getUsdUpAxis();

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

        bool shouldReorient;
        anchorApi.GetAdjustOrientationForGlobeWhenMovingAttr().Get(&shouldReorient);
        bool updateOccurred = globeAnchor->updateByUsdTransform(origin, shouldReorient);

        // We need to short circuit if an update isn't necessary.
        if (!updateOccurred) {
            return;
        }
    } else {
        auto anchorToFixed = UsdUtil::computeUsdLocalToEcefTransformForPrim(origin, anchorApi.GetPath());
        globeAnchor = GlobeAnchorRegistry::getInstance().createAnchor(anchorApi.GetPath(), anchorToFixed);
    }

    auto fixedTransform = UsdUtil::glmToUsdMatrixDecomposed(globeAnchor->getAnchorToFixedTransform());

    if (!maybeGlobeAnchor.has_value()) {
        auto localTransform = globeAnchor->getAnchorToLocalTransform(origin);
        UsdUtil::addOrUpdateTransformOpForAnchor(anchorApi.GetPath(), localTransform);
    }

    std::optional<CesiumGeospatial::Cartographic> cartographicPosition =
        CesiumGeospatial::Ellipsoid::WGS84.cartesianToCartographic(UsdUtil::usdToGlmVector(fixedTransform.position));

    anchorApi.GetPositionAttr().Set(fixedTransform.position);
    anchorApi.GetRotationAttr().Set(pxr::GfVec3d(UsdUtil::getEulerAnglesFromQuaternion(fixedTransform.orientation)));
    anchorApi.GetScaleAttr().Set(pxr::GfVec3d(fixedTransform.scale));

    if (cartographicPosition) {
        anchorApi.GetLatitudeAttr().Set(glm::degrees(cartographicPosition->latitude));
        anchorApi.GetLongitudeAttr().Set(glm::degrees(cartographicPosition->longitude));
        anchorApi.GetHeightAttr().Set(cartographicPosition->height);
    } else {
        anchorApi.GetLatitudeAttr().Set(0.0);
        anchorApi.GetLongitudeAttr().Set(0.0);
        anchorApi.GetHeightAttr().Set(0.0);
    }
}

void updateAnchorByLatLongHeight(
    [[maybe_unused]] const CesiumGeospatial::Cartographic& origin,
    [[maybe_unused]] const pxr::CesiumGlobeAnchorAPI& anchor) {}

void updateAnchorByFixedTransform(
    const CesiumGeospatial::Cartographic& origin,
    const pxr::CesiumGlobeAnchorAPI& anchorApi) {
    std::optional<std::shared_ptr<OmniGlobeAnchor>> maybeGlobeAnchor =
        GlobeAnchorRegistry::getInstance().getAnchor(anchorApi.GetPath());

    if (!maybeGlobeAnchor.has_value()) {
        // TODO: Log an error. Something bad has occurred.
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

    bool shouldReorient;
    anchorApi.GetAdjustOrientationForGlobeWhenMovingAttr().Get(&shouldReorient);
    bool updateOccurred =
        globeAnchor->updateByFixedTransform(ecefPositionVec, ecefRotationVec, ecefScaleVec, shouldReorient);

    // We need to short circuit if an update isn't necessary.
    if (!updateOccurred) {
        return;
    }

    auto localTransform = globeAnchor->getAnchorToLocalTransform(origin);
    UsdUtil::addOrUpdateTransformOpForAnchor(anchorApi.GetPath(), localTransform);

    auto cartographicPosition = globeAnchor->getCartographicPosition();

    if (cartographicPosition) {
        anchorApi.GetLatitudeAttr().Set(glm::degrees(cartographicPosition->latitude));
        anchorApi.GetLongitudeAttr().Set(glm::degrees(cartographicPosition->longitude));
        anchorApi.GetHeightAttr().Set(cartographicPosition->height);
    } else {
        // TODO: Log an error. Probably.
        anchorApi.GetLatitudeAttr().Set(0.0);
        anchorApi.GetLongitudeAttr().Set(0.0);
        anchorApi.GetHeightAttr().Set(0.0);
    }
}

} // namespace cesium::omniverse::GeospatialUtil
