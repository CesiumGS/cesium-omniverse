#pragma once

#include <CesiumGeospatial/Cartographic.h>
#include <CesiumGeospatial/GlobeAnchor.h>
#include <glm/ext/matrix_double4x4.hpp>
#include <pxr/usd/sdf/path.h>

namespace cesium::omniverse {

class OmniGlobeAnchor {
  public:
    OmniGlobeAnchor(const pxr::SdfPath& anchorPrimPath, const glm::dmat4 anchorToFixed);

    std::shared_ptr<CesiumGeospatial::GlobeAnchor> getAnchor();
    void updateByUsdTransform(const CesiumGeospatial::Cartographic& origin, bool shouldReorient);

  private:
    pxr::SdfPath _anchorPrimPath;
    std::shared_ptr<CesiumGeospatial::GlobeAnchor> _anchor;
};

} // namespace cesium::omniverse
