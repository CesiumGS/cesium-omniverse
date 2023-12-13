#include "cesium/omniverse/OmniGlobeAnchor.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/GeospatialUtil.h"
#include "cesium/omniverse/OmniGeoreference.h"
#include "cesium/omniverse/UsdUtil.h"

#include <glm/ext/matrix_relational.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/matrix_decompose.hpp>

#include <utility>

namespace cesium::omniverse {

OmniGlobeAnchor::OmniGlobeAnchor(const pxr::SdfPath& primPath)
    : _primPath(primPath) {

    const auto georeferencePath = getGeoreferencePath();
    assert(!georeferencePath.IsEmpty());
    const auto georeference = OmniGeoreference(georeferencePath);
    const auto origin = georeference.getCartographic();

    const auto geographicCoordinates = getGeographicCoordinates();

    if (geographicCoordinates == glm::dvec3(0.0, 0.0, 10.0)) {
        // Default geo coordinates. Place based on current USD position.
        const auto anchorToFixed = UsdUtil::computeUsdLocalToEcefTransformForPrim(georeferencePath, _primPath);
        _anchor = std::make_shared<CesiumGeospatial::GlobeAnchor>(anchorToFixed);

        const auto fixedTransform = UsdUtil::glmToUsdMatrixDecomposed(_anchor->getAnchorToFixedTransform());
        const auto localTransform = _anchor->getAnchorToLocalTransform(GeospatialUtil::getCoordinateSystem(origin));

        UsdUtil::addOrUpdateTransformOpForAnchor(_primPath, localTransform);

        const auto globeAnchorApi = UsdUtil::getCesiumGlobeAnchor(_primPath);

        globeAnchorApi.GetPositionAttr().Set(fixedTransform.position);
        globeAnchorApi.GetRotationAttr().Set(
            pxr::GfVec3d(UsdUtil::getEulerAnglesFromQuaternion(fixedTransform.orientation)));
        globeAnchorApi.GetScaleAttr().Set(pxr::GfVec3d(fixedTransform.scale));

        const auto cartographicPosition = Context::instance().getEllipsoid().cartesianToCartographic(
            UsdUtil::usdToGlmVector(fixedTransform.position));

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

    } else {
        // Provided geo coordinates. Place at correct location.
        const auto anchorToFixed = UsdUtil::computeUsdLocalToEcefTransformForPrim(georeferencePath, _primPath);
        _anchor = std::make_shared<CesiumGeospatial::GlobeAnchor>(anchorToFixed);

        updateAnchorByLatLongHeight(origin);
    }
}

bool OmniGlobeAnchor::getAdjustOrientationForGlobeWhenMoving() const {
    const auto globeAnchor = UsdUtil::getCesiumGlobeAnchor(_primPath);

    bool shouldReorient;
    globeAnchor.GetAdjustOrientationForGlobeWhenMovingAttr().Get(&shouldReorient);

    return shouldReorient;
}

bool OmniGlobeAnchor::getDetectTransformChanges() const {
    const auto globeAnchor = UsdUtil::getCesiumGlobeAnchor(_primPath);

    bool detectTransformChanges;
    globeAnchor.GetDetectTransformChangesAttr().Get(&detectTransformChanges);

    return detectTransformChanges;
}

pxr::SdfPath OmniGlobeAnchor::getGeoreferencePath() const {
    const auto globeAnchor = UsdUtil::getCesiumGlobeAnchor(_primPath);

    pxr::SdfPathVector targets;
    globeAnchor.GetGeoreferenceBindingRel().GetForwardedTargets(&targets);

    if (targets.empty()) {
        return {};
    }

    auto georeferencePath = targets.front();

    if (!UsdUtil::isCesiumGeoreference(georeferencePath)) {
        return {};
    }

    return georeferencePath;
}

const pxr::GfMatrix4d& OmniGlobeAnchor::getCachedTransformation() const {
    return _valueCache.transformation;
}

const pxr::GfVec3d& OmniGlobeAnchor::getCachedGeographicCoordinate() const {
    return _valueCache.geographicCoordinate;
}

const pxr::GfVec3d& OmniGlobeAnchor::getCachedEcefPosition() const {
    return _valueCache.ecefPosition;
}

const pxr::GfVec3d& OmniGlobeAnchor::getCachedEcefRotation() const {
    return _valueCache.ecefRotation;
}

const pxr::GfVec3d& OmniGlobeAnchor::getCachedEcefScale() const {
    return _valueCache.ecefScale;
}

void OmniGlobeAnchor::updateCachedValues() {
    auto globeAnchorAPI = UsdUtil::getCesiumGlobeAnchor(_primPath);

    pxr::GfVec3d newGeographicCoordinate;
    globeAnchorAPI.GetGeographicCoordinateAttr().Get(&newGeographicCoordinate);
    _valueCache.geographicCoordinate = newGeographicCoordinate;

    pxr::GfVec3d newEcefPosition;
    globeAnchorAPI.GetPositionAttr().Get(&newEcefPosition);
    _valueCache.ecefPosition = newEcefPosition;

    pxr::GfVec3d newEcefRotation;
    globeAnchorAPI.GetRotationAttr().Get(&newEcefRotation);
    _valueCache.ecefRotation = newEcefRotation;

    pxr::GfVec3d newEcefScale;
    globeAnchorAPI.GetScaleAttr().Get(&newEcefScale);
    _valueCache.ecefScale = newEcefScale;

    auto maybeNewTransform = UsdUtil::getCesiumTransformOpValueForPathIfExists(_primPath);
    if (maybeNewTransform.has_value()) {
        _valueCache.transformation = maybeNewTransform.value();
    }
}

const glm::dmat4& OmniGlobeAnchor::getAnchorToFixedTransform() const {
    return _anchor->getAnchorToFixedTransform();
}

const glm::dmat4 OmniGlobeAnchor::getAnchorToLocalTransform(const CesiumGeospatial::Cartographic& origin) {
    return _anchor->getAnchorToLocalTransform(GeospatialUtil::getCoordinateSystem(origin));
}

std::optional<CesiumGeospatial::Cartographic> OmniGlobeAnchor::getCartographicPosition() {
    glm::dvec3 fs{};
    glm::dquat fr{};
    glm::dvec3 ft{};
    glm::dvec3 fskew{};
    glm::dvec4 fperspective{};

    [[maybe_unused]] auto fixedTransform = glm::decompose(getAnchorToFixedTransform(), fs, fr, ft, fskew, fperspective);
    assert(fixedTransform);

    return CesiumGeospatial::Ellipsoid::WGS84.cartesianToCartographic(ft);
}

const pxr::SdfPath& OmniGlobeAnchor::getPrimPath() const {
    return _primPath;
}

void OmniGlobeAnchor::updateByFixedTransform(
    const glm::dvec3& ecefPositionVec,
    const glm::dvec3& ecefRotationRadVec,
    const glm::dvec3& ecefScaleVec,
    bool shouldReorient) {
    auto translation = glm::translate(glm::dmat4(1.0), ecefPositionVec);
    auto rotation = glm::eulerAngleXYZ<double>(ecefRotationRadVec.x, ecefRotationRadVec.y, ecefRotationRadVec.z);
    auto scale = glm::scale(glm::dmat4(1.0), ecefScaleVec);
    auto newAnchorToFixed = translation * rotation * scale;

    _anchor->setAnchorToFixedTransform(newAnchorToFixed, shouldReorient, Context::instance().getEllipsoid());
}

void OmniGlobeAnchor::updateByGeographicCoordinates(CesiumGeospatial::Cartographic& cartographic, bool shouldReorient) {
    auto newEcefPositionVec = CesiumGeospatial::Ellipsoid::WGS84.cartographicToCartesian(cartographic);

    auto ecefRotationDegVec = UsdUtil::usdToGlmVector(_valueCache.ecefRotation);
    auto ecefScaleVec = UsdUtil::usdToGlmVector(_valueCache.ecefScale);

    updateByFixedTransform(newEcefPositionVec, glm::radians(ecefRotationDegVec), ecefScaleVec, shouldReorient);
}

void OmniGlobeAnchor::updateByUsdTransform(const CesiumGeospatial::Cartographic& origin, bool shouldReorient) {
    const auto newAnchorToFixed = UsdUtil::computeUsdLocalToEcefTransformForPrim(origin, _primPath);

    _anchor->setAnchorToFixedTransform(newAnchorToFixed, shouldReorient, Context::instance().getEllipsoid());
}

} // namespace cesium::omniverse
