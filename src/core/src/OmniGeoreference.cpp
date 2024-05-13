#include "cesium/omniverse/OmniGeoreference.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/UsdUtil.h"

#include <CesiumGeospatial/Ellipsoid.h>
#include <CesiumGeospatial/LocalHorizontalCoordinateSystem.h>
#include <CesiumUsdSchemas/georeference.h>
#include <glm/glm.hpp>
#include <pxr/usd/usdGeom/tokens.h>

namespace cesium::omniverse {

OmniGeoreference::OmniGeoreference(Context* pContext, const pxr::SdfPath& path)
    : _pContext(pContext)
    , _path(path)
    , _ellipsoid(CesiumGeospatial::Ellipsoid::WGS84) {
    update();
}

const pxr::SdfPath& OmniGeoreference::getPath() const {
    return _path;
}

CesiumGeospatial::Cartographic OmniGeoreference::getOrigin() const {
    const auto cesiumGeoreference = UsdUtil::getCesiumGeoreference(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumGeoreference)) {
        return {0.0, 0.0, 0.0};
    }

    double longitude;
    double latitude;
    double height;

    cesiumGeoreference.GetGeoreferenceOriginLongitudeAttr().Get(&longitude);
    cesiumGeoreference.GetGeoreferenceOriginLatitudeAttr().Get(&latitude);
    cesiumGeoreference.GetGeoreferenceOriginHeightAttr().Get(&height);

    return {glm::radians(longitude), glm::radians(latitude), height};
}

const CesiumGeospatial::Ellipsoid& OmniGeoreference::getEllipsoid() const {
    return _ellipsoid;
}

CesiumGeospatial::LocalHorizontalCoordinateSystem OmniGeoreference::getLocalCoordinateSystem() const {
    const auto origin = getOrigin();

    const auto upAxis = UsdUtil::getUsdUpAxis(_pContext->getUsdStage());
    const auto scaleInMeters = UsdUtil::getUsdMetersPerUnit(_pContext->getUsdStage());

    if (upAxis == pxr::UsdGeomTokens->z) {
        return {
            origin,
            CesiumGeospatial::LocalDirection::East,
            CesiumGeospatial::LocalDirection::North,
            CesiumGeospatial::LocalDirection::Up,
            scaleInMeters,
            _ellipsoid,
        };
    }

    return {
        origin,
        CesiumGeospatial::LocalDirection::East,
        CesiumGeospatial::LocalDirection::Up,
        CesiumGeospatial::LocalDirection::South,
        scaleInMeters,
        _ellipsoid,
    };
}

void OmniGeoreference::update() {
    const auto cesiumGeoreference = UsdUtil::getCesiumGeoreference(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumGeoreference)) {
        return;
    }

    const auto ecefToUsdTransform = getLocalCoordinateSystem().getEcefToLocalTransformation();

    cesiumGeoreference.GetEcefToUsdTransformAttr().Set(UsdUtil::glmToUsdMatrix(ecefToUsdTransform));
}

} // namespace cesium::omniverse
