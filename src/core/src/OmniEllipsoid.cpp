#include "cesium/omniverse/OmniEllipsoid.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/UsdUtil.h"

#include <CesiumGeospatial/Ellipsoid.h>
#include <CesiumUsdSchemas/ellipsoid.h>

namespace cesium::omniverse {

OmniEllipsoid::OmniEllipsoid(Context* pContext, const pxr::SdfPath& path)
    : _pContext(pContext)
    , _path(path) {}

const pxr::SdfPath& OmniEllipsoid::getPath() const {
    return _path;
}

CesiumGeospatial::Ellipsoid OmniEllipsoid::getEllipsoid() const {
    const auto cesiumEllipsoid = UsdUtil::getCesiumEllipsoid(_pContext->getUsdStage(), _path);

    glm::dvec3 radii(0.0);

    if (UsdUtil::isSchemaValid(cesiumEllipsoid)) {
        pxr::GfVec3d radiiUsd;
        cesiumEllipsoid.GetRadiiAttr().Get(&radiiUsd);
        radii = UsdUtil::usdToGlmVector(radiiUsd);
    }

    return {radii};
}

} // namespace cesium::omniverse
