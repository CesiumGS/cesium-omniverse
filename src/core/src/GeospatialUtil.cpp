#include "cesium/omniverse/GeospatialUtil.h"

namespace cesium::omniverse::GeospatialUtil {

CesiumGeospatial::Cartographic convertGeoreferenceToCartographic(const pxr::CesiumGeoreference& georeference) {
    double longitude;
    double latitude;
    double height;
    georeference.GetGeoreferenceOriginLongitudeAttr().Get<double>(&longitude);
    georeference.GetGeoreferenceOriginLatitudeAttr().Get<double>(&latitude);
    georeference.GetGeoreferenceOriginHeightAttr().Get<double>(&height);

    return CesiumGeospatial::Cartographic(glm::radians(longitude), glm::radians(latitude), height);
}

[[maybe_unused]] CesiumGeospatial::LocalHorizontalCoordinateSystem
getCoordinateSystem(const pxr::CesiumGeoreference& georeference, const double scaleInMeters) {
    auto cartographicOrigin = GeospatialUtil::convertGeoreferenceToCartographic(georeference);
    return CesiumGeospatial::LocalHorizontalCoordinateSystem(
        cartographicOrigin,
        CesiumGeospatial::LocalDirection::East,
        CesiumGeospatial::LocalDirection::Up,
        CesiumGeospatial::LocalDirection::South,
        scaleInMeters);
}

} // namespace cesium::omniverse::GeospatialUtil
