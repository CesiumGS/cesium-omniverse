#include "cesium/omniverse/OmniGlobeAnchor.h"

#include "cesium/omniverse/GeospatialUtil.h"
#include "cesium/omniverse/UsdUtil.h"

#include <glm/ext/matrix_relational.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/matrix_decompose.hpp>

#include <utility>

namespace cesium::omniverse {

OmniGlobeAnchor::OmniGlobeAnchor(pxr::SdfPath anchorPrimPath, const glm::dmat4 anchorToFixed)
    : _anchorPrimPath{std::move(anchorPrimPath)} {
    _anchor = std::make_shared<CesiumGeospatial::GlobeAnchor>(anchorToFixed);
}

const glm::dmat4 OmniGlobeAnchor::getAnchorToFixedTransform() {
    return _anchor->getAnchorToFixedTransform();
}

const glm::dmat4 OmniGlobeAnchor::getAnchorToLocalTransform(const CesiumGeospatial::Cartographic& origin) {
    return _anchor->getAnchorToLocalTransform(
        GeospatialUtil::getCoordinateSystem(origin, UsdUtil::getUsdMetersPerUnit()));
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

[[maybe_unused]] const pxr::SdfPath OmniGlobeAnchor::getPrimPath() {
    return _anchorPrimPath;
}

bool OmniGlobeAnchor::updateByFixedTransform(
    glm::dvec3 ecefPositionVec,
    glm::dvec3 ecefRotationVec,
    glm::dvec3 ecefScaleVec,
    bool shouldReorient) {
    auto translation = glm::translate(glm::dmat4(1.0), ecefPositionVec);
    auto rotation = glm::eulerAngleYXZ<double>(
        glm::radians(ecefRotationVec.y), glm::radians(ecefRotationVec.x), glm::radians(ecefRotationVec.z));
    auto scale = glm::scale(glm::dmat4(1.0), ecefScaleVec);
    auto newAnchorToFixed = translation * rotation * scale;

    // Epsilon needs to be so large to prevent some weird UX jitter from occurring.
    const double epsilon = 0.01;

    const auto anchorToFixed = _anchor->getAnchorToFixedTransform();
    if (glm::all(glm::equal(anchorToFixed, newAnchorToFixed, epsilon))) {
        return false;
    }

    _anchor->setAnchorToFixedTransform(newAnchorToFixed, shouldReorient);

    return true;
}

bool OmniGlobeAnchor::updateByUsdTransform(const CesiumGeospatial::Cartographic& origin, bool shouldReorient) {
    const auto newAnchorToFixed = UsdUtil::computeUsdLocalToEcefTransformForPrim(origin, _anchorPrimPath);

    // Epsilon needs to be so large to prevent some weird UX jitter from occurring.
    const double epsilon = 0.01;

    const auto anchorToFixed = _anchor->getAnchorToFixedTransform();
    if (glm::all(glm::equal(anchorToFixed, newAnchorToFixed, epsilon))) {
        return false;
    }

    _anchor->setAnchorToFixedTransform(newAnchorToFixed, shouldReorient);

    return true;
}

} // namespace cesium::omniverse
