#include "cesium/omniverse/GeospatialUtil.h"

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

} // namespace cesium::omniverse::GeospatialUtil
