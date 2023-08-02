#include "cesium/omniverse/OmniGlobeAnchor.h"

#include "cesium/omniverse/GeospatialUtil.h"
#include "cesium/omniverse/UsdUtil.h"

namespace cesium::omniverse {

OmniGlobeAnchor::OmniGlobeAnchor(const pxr::SdfPath& anchorPrimPath, const glm::dmat4 anchorToFixed)
    : _anchorPrimPath{anchorPrimPath} {
    _anchor = std::make_shared<CesiumGeospatial::GlobeAnchor>(anchorToFixed);
}

std::shared_ptr<CesiumGeospatial::GlobeAnchor> OmniGlobeAnchor::getAnchor() {
    return _anchor;
}

void OmniGlobeAnchor::updateByUsdTransform(const CesiumGeospatial::Cartographic& origin, bool shouldReorient) {
    const auto anchorToFixed = UsdUtil::computeUsdLocalToEcefTransformForPrim(origin, _anchorPrimPath);
    _anchor->setAnchorToFixedTransform(anchorToFixed, shouldReorient);
}

} // namespace cesium::omniverse
