#include "cesium/omniverse/OmniGeoreference.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/UsdUtil.h"

#include <CesiumGeospatial/Cartographic.h>
#include <CesiumUsdSchemas/georeference.h>

namespace cesium::omniverse {

OmniGeoreference::OmniGeoreference(const pxr::SdfPath& path)
    : _path(path) {}

pxr::SdfPath OmniGeoreference::getPath() const {
    return _path;
}

CesiumGeospatial::Cartographic OmniGeoreference::getCartographic() const {
    auto georeference = UsdUtil::getCesiumGeoreference(_path);

    double longitude;
    double latitude;
    double height;

    georeference.GetGeoreferenceOriginLongitudeAttr().Get<double>(&longitude);
    georeference.GetGeoreferenceOriginLatitudeAttr().Get<double>(&latitude);
    georeference.GetGeoreferenceOriginHeightAttr().Get<double>(&height);

    longitude = glm::radians(longitude);
    latitude = glm::radians(latitude);

    return {longitude, latitude, height};
}

void OmniGeoreference::setCartographic(const CesiumGeospatial::Cartographic& cartographic) const {
    auto georeference = UsdUtil::getCesiumGeoreference(_path);

    georeference.GetGeoreferenceOriginLongitudeAttr().Set<double>(glm::degrees(cartographic.longitude));
    georeference.GetGeoreferenceOriginLatitudeAttr().Set<double>(glm::degrees(cartographic.latitude));
    georeference.GetGeoreferenceOriginHeightAttr().Set<double>(cartographic.height);
}

} // namespace cesium::omniverse
