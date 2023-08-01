#include "cesium/omniverse/GeospatialUtil.h"

#include "cesium/omniverse/GlobeAnchorRegistry.h"
#include "cesium/omniverse/UsdUtil.h"

#include <CesiumGeometry/Transforms.h>
#include <CesiumGeospatial/GlobeAnchor.h>
#include <CesiumGeospatial/GlobeTransforms.h>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/xformCommonAPI.h>

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

glm::dmat4 getAxisConversionTransform() {
    const auto upAxis = UsdUtil::getUsdUpAxis();

    auto axisConversion = glm::dmat4(1.0);

    // USD up axis can be either Y or Z
    if (upAxis == pxr::UsdGeomTokens->y) {
        axisConversion = CesiumGeometry::Transforms::Y_UP_TO_Z_UP;
    }

    return axisConversion;
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

glm::dmat4 getEastNorthUpToFixedFrame(const CesiumGeospatial::Cartographic& cartographic) {
    const auto cartesian = CesiumGeospatial::Ellipsoid::WGS84.cartographicToCartesian(cartographic);
    const auto matrix = CesiumGeospatial::GlobeTransforms::eastNorthUpToFixedFrame(cartesian);
    return matrix;
}

glm::dmat4 getUnitConversionTransform() {
    const auto metersPerUnit = UsdUtil::getUsdMetersPerUnit();
    const auto matrix = glm::scale(glm::dmat4(1.0), glm::dvec3(metersPerUnit));
    return matrix;
}

void updateAnchorByUsdTransform(const CesiumGeospatial::Cartographic& origin, const pxr::CesiumGlobeAnchorAPI& anchor) {
    auto ecefTransform = UsdUtil::computeUsdToEcefTransformForPrim(origin, anchor.GetPath());

    std::optional<std::shared_ptr<CesiumGeospatial::GlobeAnchor>> maybeGlobeAnchor =
        GlobeAnchorRegistry::getInstance().getAnchor(anchor.GetPath());

    std::shared_ptr<CesiumGeospatial::GlobeAnchor> globeAnchor;
    if (maybeGlobeAnchor.has_value()) {
        globeAnchor = maybeGlobeAnchor.value();

        bool shouldReorient;
        anchor.GetAdjustOrientationForGlobeWhenMovingAttr().Get(&shouldReorient);
        globeAnchor->setAnchorToFixedTransform(ecefTransform, shouldReorient);
    } else {
        globeAnchor = GlobeAnchorRegistry::getInstance().createAnchor(anchor.GetPath(), ecefTransform);
    }

    auto fixedTransform = UsdUtil::glmToUsdMatrixDecomposed(globeAnchor->getAnchorToFixedTransform());

    if (!maybeGlobeAnchor.has_value()) {
        // We need to do this when it's a new anchor.

        auto xformCommonApi = pxr::UsdGeomXformCommonAPI(anchor.GetPrim());
        xformCommonApi.SetTranslate(pxr::GfVec3f(0.f, 0.f, 0.f));
        xformCommonApi.SetRotate(pxr::GfVec3f(0.0f, 0.0f, 0.0f));
        xformCommonApi.SetScale(pxr::GfVec3f(1.f, 1.f, 1.f));

        auto localTransform = globeAnchor->getAnchorToLocalTransform(
            GeospatialUtil::getCoordinateSystem(origin, UsdUtil::getUsdMetersPerUnit()));
        auto xform = pxr::UsdGeomXform(anchor.GetPrim());
        xform.AddTransformOp().Set(UsdUtil::glmToUsdMatrix(localTransform));
    }

    std::optional<CesiumGeospatial::Cartographic> cartographicPosition =
        CesiumGeospatial::Ellipsoid::WGS84.cartesianToCartographic(UsdUtil::usdToGlmVector(fixedTransform.position));

    anchor.GetPositionAttr().Set(fixedTransform.position);
    anchor.GetRotationAttr().Set(pxr::GfVec3d(UsdUtil::getEulerAnglesFromQuaternion(fixedTransform.orientation)));
    anchor.GetScaleAttr().Set(pxr::GfVec3d(fixedTransform.scale));

    if (cartographicPosition) {
        anchor.GetLatitudeAttr().Set(glm::degrees(cartographicPosition->latitude));
        anchor.GetLongitudeAttr().Set(glm::degrees(cartographicPosition->longitude));
        anchor.GetHeightAttr().Set(cartographicPosition->height);
    } else {
        anchor.GetLatitudeAttr().Set(0.0);
        anchor.GetLongitudeAttr().Set(0.0);
        anchor.GetHeightAttr().Set(0.0);
    }
}

void updateAnchorByLatLongHeight(
    [[maybe_unused]] const CesiumGeospatial::Cartographic& origin,
    [[maybe_unused]] const pxr::CesiumGlobeAnchorAPI& anchor) {}

void updateAnchorByFixedTransform(
    const CesiumGeospatial::Cartographic& origin,
    const pxr::CesiumGlobeAnchorAPI& anchor) {
    std::optional<std::shared_ptr<CesiumGeospatial::GlobeAnchor>> maybeGlobeAnchor =
        GlobeAnchorRegistry::getInstance().getAnchor(anchor.GetPath());

    if (!maybeGlobeAnchor.has_value()) {
        // TODO: Log an error. Something bad has occurred.
        return;
    }

    std::shared_ptr<CesiumGeospatial::GlobeAnchor> globeAnchor = maybeGlobeAnchor.value();

    pxr::GfVec3d usdEcefPositionVec;
    anchor.GetPositionAttr().Get(&usdEcefPositionVec);
    pxr::GfVec3d usdEcefRotationVec;
    anchor.GetRotationAttr().Get(&usdEcefRotationVec);
    pxr::GfVec3d usdEcefScaleVec;
    anchor.GetScaleAttr().Get(&usdEcefScaleVec);

    // TODO: Move the code to create the transformation matrix to a helper function.
    auto ecefPositionVec = UsdUtil::usdToGlmVector(usdEcefPositionVec);
    auto ecefRotationVec = UsdUtil::usdToGlmVector(usdEcefRotationVec);
    auto ecefScaleVec = UsdUtil::usdToGlmVector(usdEcefScaleVec);

    auto translation = glm::translate(glm::dmat4(1.0), ecefPositionVec);
    auto rotation = glm::eulerAngleYXZ<double>(
        glm::radians(ecefRotationVec.y), glm::radians(ecefRotationVec.x), glm::radians(ecefRotationVec.z));
    auto scale = glm::scale(glm::dmat4(1.0), ecefScaleVec);
    auto transformationMatrix = translation * rotation * scale;

    bool shouldReorient;
    anchor.GetAdjustOrientationForGlobeWhenMovingAttr().Get(&shouldReorient);
    globeAnchor->setAnchorToFixedTransform(transformationMatrix, shouldReorient);

    glm::dvec3 ls{};
    glm::dquat lr{};
    glm::dvec3 lt{};
    glm::dvec3 lskew{};
    glm::dvec4 lperspective{};

    [[maybe_unused]] auto localTransform = glm::decompose(
        globeAnchor->getAnchorToLocalTransform(
            GeospatialUtil::getCoordinateSystem(origin, UsdUtil::getUsdMetersPerUnit())),
        ls,
        lr,
        lt,
        lskew,
        lperspective);
    assert(localTransform);

    auto xform = pxr::UsdGeomXformCommonAPI(anchor.GetPrim());
    xform.SetTranslate(UsdUtil::glmToUsdVector(lt));
    xform.SetRotate(UsdUtil::getEulerAnglesFromQuaternion(pxr::GfQuatf(UsdUtil::glmToUsdQuat(lr))));
    xform.SetScale(pxr::GfVec3f(UsdUtil::glmToUsdVector(ls)));

    glm::dvec3 fs{};
    glm::dquat fr{};
    glm::dvec3 ft{};
    glm::dvec3 fskew{};
    glm::dvec4 fperspective{};

    [[maybe_unused]] auto fixedTransform =
        glm::decompose(globeAnchor->getAnchorToFixedTransform(), fs, fr, ft, fskew, fperspective);
    assert(fixedTransform);

    auto cartographicPosition = CesiumGeospatial::Ellipsoid::WGS84.cartesianToCartographic(ft);

    if (cartographicPosition) {
        anchor.GetLatitudeAttr().Set(glm::degrees(cartographicPosition->latitude));
        anchor.GetLongitudeAttr().Set(glm::degrees(cartographicPosition->longitude));
        anchor.GetHeightAttr().Set(cartographicPosition->height);
    } else {
        // TODO: Log an error. Probably.
        anchor.GetLatitudeAttr().Set(0.0);
        anchor.GetLongitudeAttr().Set(0.0);
        anchor.GetHeightAttr().Set(0.0);
    }
}

} // namespace cesium::omniverse::GeospatialUtil
