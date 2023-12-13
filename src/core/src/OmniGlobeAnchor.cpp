#include "cesium/omniverse/OmniGlobeAnchor.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/LoggerSink.h"
#include "cesium/omniverse/OmniGeoreference.h"
#include "cesium/omniverse/UsdUtil.h"

#include <CesiumGeospatial/Cartographic.h>
#include <CesiumGeospatial/Ellipsoid.h>
#include <CesiumUtility/Math.h>
#include <glm/ext/matrix_relational.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/matrix_decompose.hpp>

#include <utility>

namespace cesium::omniverse {

namespace {
const auto DEFAULT_GEOGRAPHIC_COORDINATES = CesiumGeospatial::Cartographic(0.0, 0.0, 10.0);
}

bool equal(const CesiumGeospatial::Cartographic& a, const CesiumGeospatial::Cartographic& b) {
    const auto& aVec = *reinterpret_cast<const glm::dvec3*>(&a);
    const auto& bVec = *reinterpret_cast<const glm::dvec3*>(&b);
    return aVec == bVec;
}

bool epsilonEqual(const CesiumGeospatial::Cartographic& a, const CesiumGeospatial::Cartographic& b, double epsilon) {
    const auto& aVec = *reinterpret_cast<const glm::dvec3*>(&a);
    const auto& bVec = *reinterpret_cast<const glm::dvec3*>(&b);
    return glm::all(glm::epsilonEqual(aVec, bVec, epsilon));
}

bool epsilonEqual(const glm::dmat4& a, const glm::dmat4& b, double epsilon) {
    return glm::all(glm::epsilonEqual(a[0], b[0], epsilon)) && glm::all(glm::epsilonEqual(a[1], b[1], epsilon)) &&
           glm::all(glm::epsilonEqual(a[2], b[2], epsilon)) && glm::all(glm::epsilonEqual(a[3], b[3], epsilon));
}

bool epsilonEqual(const glm::dvec3& a, const glm::dvec3& b, double epsilon) {
    return glm::all(glm::epsilonEqual(a, b, epsilon));
}

struct Decomposed {
    glm::dvec3 translation;
    glm::dvec3 rotation;
    glm::dvec3 scale;
};

Decomposed decompose(const glm::dmat4& matrix) {
    glm::dvec3 scale;
    glm::dquat rotation;
    glm::dvec3 translation;
    glm::dvec3 skew;
    glm::dvec4 perspective;

    [[maybe_unused]] const auto decomposable = glm::decompose(matrix, scale, rotation, translation, skew, perspective);
    assert(decomposable);

    return {translation, glm::eulerAngles(rotation), scale};
}

OmniGlobeAnchor::OmniGlobeAnchor(const pxr::SdfPath& primPath, const CesiumGeospatial::Ellipsoid& ellipsoid)
    : _primPath(primPath)
    , _ellipsoid(ellipsoid) {

    const auto georeferencePath = getGeoreferencePath();
    const auto anchorToFixed = UsdUtil::computeUsdLocalToEcefTransformForPrim(georeferencePath, _primPath);
    _anchor = std::make_shared<CesiumGeospatial::GlobeAnchor>(anchorToFixed);

    updateCachedValues();

    if (equal(getGeographicCoordinates(), DEFAULT_GEOGRAPHIC_COORDINATES)) {
        // Default geo coordinates. Place based on current USD position.
        updateTransformOp();
        updateEcefPositionRotationScale();
        updateGeographicCoordinates();
        updateCachedValues();

    } else {
        // Provided geo coordinates. Place at correct location.
        updateEcefPositionRotationScale();
        updateByGeographicCoordinatesInternal();
        updateTransformOp();
        updateEcefPositionRotationScale();
        updateCachedValues();
    }
}

void OmniGlobeAnchor::updateByGeographicCoordinates() {
    const auto geographicCoordinates = getGeographicCoordinates();

    const auto tolerance = CesiumUtility::Math::Epsilon7;
    if (epsilonEqual(geographicCoordinates, _cachedGeographicCoordinates, tolerance)) {
        // Short circuit if we don't need to do an actual update.
        return;
    }

    updateByGeographicCoordinatesInternal();
    updateTransformOp();
    updateEcefPositionRotationScale();
    updateCachedValues();
}

void OmniGlobeAnchor::updateByLocalTransform() {
    const auto localTransform = UsdUtil::getCesiumTransformOpValueForPathIfExists(_primPath);

    const auto tolerance = CesiumUtility::Math::Epsilon2;
    if (localTransform.has_value() &&
        epsilonEqual(UsdUtil::usdToGlmMatrix(localTransform.value()), _cachedLocalTransform, tolerance)) {
        // Short circuit if we don't need to do an actual update.
        return;
    }

    updateByLocalTransformInternal();
    updateEcefPositionRotationScale();
    updateGeographicCoordinates();
    updateCachedValues();
}

void OmniGlobeAnchor::updateByFixedTransform() {
    const auto ecefPosition = getEcefPosition();
    const auto ecefRotation = getEcefRotation();
    const auto ecefScale = getEcefScale();

    const auto tolerance = CesiumUtility::Math::Epsilon4;
    if (epsilonEqual(ecefPosition, _cachedEcefPosition, tolerance) &&
        epsilonEqual(ecefRotation, _cachedEcefRotation, tolerance) &&
        epsilonEqual(ecefScale, _cachedEcefScale, tolerance)) {

        // Short circuit if we don't need to do an actual update.
        return;
    }

    updateByFixedTransformInternal(ecefPosition, ecefRotation, ecefScale);
    updateTransformOp();
    updateGeographicCoordinates();
    updateCachedValues();
}

void OmniGlobeAnchor::updateOrigin() {
    const auto anchorToLocal = getAnchorToLocalTransform();
    UsdUtil::addOrUpdateTransformOpForAnchor(_primPath, anchorToLocal);
    updateCachedValues();
}

bool OmniGlobeAnchor::getAdjustOrientation() const {
    const auto globeAnchor = UsdUtil::getCesiumGlobeAnchor(_primPath);

    bool adjustOrientation;
    globeAnchor.GetAdjustOrientationForGlobeWhenMovingAttr().Get(&adjustOrientation);

    return adjustOrientation;
}

bool OmniGlobeAnchor::getDetectTransformChanges() const {
    const auto globeAnchor = UsdUtil::getCesiumGlobeAnchor(_primPath);

    bool detectTransformChanges;
    globeAnchor.GetDetectTransformChangesAttr().Get(&detectTransformChanges);

    return detectTransformChanges;
}

CesiumGeospatial::Cartographic OmniGlobeAnchor::getGeographicCoordinates() const {
    const auto globeAnchor = UsdUtil::getCesiumGlobeAnchor(_primPath);

    pxr::GfVec3d coordinates;
    globeAnchor.GetGeographicCoordinateAttr().Get(&coordinates);

    const auto longitude = glm::radians(coordinates[1]);
    const auto latitude = glm::radians(coordinates[0]);
    const auto height = coordinates[2];

    return {longitude, latitude, height};
}

glm::dvec3 OmniGlobeAnchor::getEcefPosition() const {
    const auto globeAnchor = UsdUtil::getCesiumGlobeAnchor(_primPath);

    pxr::GfVec3d positionEcef;
    globeAnchor.GetPositionAttr().Get(&positionEcef);

    return UsdUtil::usdToGlmVector(positionEcef);
}

glm::dvec3 OmniGlobeAnchor::getEcefRotation() const {
    const auto globeAnchor = UsdUtil::getCesiumGlobeAnchor(_primPath);

    pxr::GfVec3d rotationEcef;
    globeAnchor.GetRotationAttr().Get(&rotationEcef);

    return UsdUtil::usdToGlmVector(rotationEcef);
}

glm::dvec3 OmniGlobeAnchor::getEcefScale() const {
    const auto globeAnchor = UsdUtil::getCesiumGlobeAnchor(_primPath);

    pxr::GfVec3d scaleEcef;
    globeAnchor.GetScaleAttr().Get(&scaleEcef);

    return UsdUtil::usdToGlmVector(scaleEcef);
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

void OmniGlobeAnchor::updateGeographicCoordinates() {
    const auto globeAnchor = UsdUtil::getCesiumGlobeAnchor(_primPath);

    const auto& anchorToFixed = _anchor->getAnchorToFixedTransform();
    const auto ecefPosition = glm::dvec3(anchorToFixed[3]);
    const auto cartographic = _ellipsoid.cartesianToCartographic(ecefPosition).value_or(DEFAULT_GEOGRAPHIC_COORDINATES);

    globeAnchor.GetGeographicCoordinateAttr().Set(
        pxr::GfVec3d(glm::degrees(cartographic.latitude), glm::degrees(cartographic.longitude), cartographic.height));
}

void OmniGlobeAnchor::updateEcefPositionRotationScale() {
    const auto globeAnchor = UsdUtil::getCesiumGlobeAnchor(_primPath);

    const auto decomposed = decompose(_anchor->getAnchorToFixedTransform());
    globeAnchor.GetPositionAttr().Set(UsdUtil::glmToUsdVector(decomposed.translation));
    globeAnchor.GetRotationAttr().Set(UsdUtil::glmToUsdVector(decomposed.rotation));
    globeAnchor.GetScaleAttr().Set(UsdUtil::glmToUsdVector(decomposed.scale));
}

void OmniGlobeAnchor::updateTransformOp() {
    UsdUtil::addOrUpdateTransformOpForAnchor(_primPath, getAnchorToLocalTransform());
}

void OmniGlobeAnchor::updateCachedValues() {
    _cachedGeographicCoordinates = getGeographicCoordinates();
    _cachedEcefPosition = getEcefPosition();
    _cachedEcefRotation = getEcefRotation();
    _cachedEcefScale = getEcefScale();

    const auto maybeNewTransform = UsdUtil::getCesiumTransformOpValueForPathIfExists(_primPath);
    if (maybeNewTransform.has_value()) {
        _cachedLocalTransform = UsdUtil::usdToGlmMatrix(maybeNewTransform.value());
    }
}

void OmniGlobeAnchor::updateByGeographicCoordinatesInternal() {
    const auto newEcefPosition = _ellipsoid.cartographicToCartesian(getGeographicCoordinates());
    updateByFixedTransformInternal(newEcefPosition, _cachedEcefRotation, _cachedEcefScale);
}

void OmniGlobeAnchor::updateByLocalTransformInternal() {
    const auto adjustOrientation = getAdjustOrientation();
    const auto anchorToFixed = UsdUtil::computeUsdLocalToEcefTransformForPrim(getGeoreferencePath(), _primPath);
    _anchor->setAnchorToFixedTransform(anchorToFixed, adjustOrientation, _ellipsoid);
}

void OmniGlobeAnchor::updateByFixedTransformInternal(
    const glm::dvec3& ecefPosition,
    const glm::dvec3& ecefRotation,
    const glm::dvec3& ecefScale) {
    const auto translation = glm::translate(glm::dmat4(1.0), ecefPosition);
    const auto rotation = glm::eulerAngleXYZ<double>(ecefRotation.x, ecefRotation.y, ecefRotation.z);
    const auto scale = glm::scale(glm::dmat4(1.0), ecefScale);
    const auto newAnchorToFixed = translation * rotation * scale;

    const auto adjustOrientation = getAdjustOrientation();
    _anchor->setAnchorToFixedTransform(newAnchorToFixed, adjustOrientation, _ellipsoid);
}

glm::dmat4 OmniGlobeAnchor::getAnchorToLocalTransform() const {
    const auto origin = getOrigin();
    const auto localCoordinateSystem = UsdUtil::getLocalCoordinateSystem(origin, _ellipsoid);
    return _anchor->getAnchorToLocalTransform(localCoordinateSystem);
}

CesiumGeospatial::Cartographic OmniGlobeAnchor::getOrigin() const {
    const auto georeferencePath = getGeoreferencePath();
    auto origin = DEFAULT_GEOGRAPHIC_COORDINATES;

    if (!georeferencePath.IsEmpty()) {
        const auto georeference = OmniGeoreference(georeferencePath);
        origin = georeference.getOrigin();
    }

    return origin;
}

} // namespace cesium::omniverse
