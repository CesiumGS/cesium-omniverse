#pragma once

#include <CesiumGeospatial/Cartographic.h>
#include <CesiumGeospatial/GlobeAnchor.h>
#include <glm/ext/matrix_double4x4.hpp>
#include <pxr/usd/sdf/path.h>

namespace cesium::omniverse {

class OmniGlobeAnchor {
  public:
    OmniGlobeAnchor(pxr::SdfPath anchorPrimPath, const glm::dmat4 anchorToFixed);

    const glm::dmat4 getAnchorToFixedTransform();
    const glm::dmat4 getAnchorToLocalTransform(const CesiumGeospatial::Cartographic& origin);
    std::optional<CesiumGeospatial::Cartographic> getCartographicPosition();
    [[maybe_unused]] const pxr::SdfPath getPrimPath();
    bool updateByFixedTransform(
        glm::dvec3 ecefPositionVec,
        glm::dvec3 ecefRotationVec,
        glm::dvec3 ecefScaleVec,
        bool shouldReorient);
    bool updateByUsdTransform(const CesiumGeospatial::Cartographic& origin, bool shouldReorient);

  private:
    pxr::SdfPath _anchorPrimPath;
    std::shared_ptr<CesiumGeospatial::GlobeAnchor> _anchor;
};

} // namespace cesium::omniverse
