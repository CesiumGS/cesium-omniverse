#include "cesium/omniverse/OmniGeoreference.h"

#include "cesium/omniverse/UsdUtil.h"

#include <CesiumGeospatial/Cartographic.h>
#include <CesiumUsdSchemas/georeference.h>

namespace cesium::omniverse {

OmniGeoreference::OmniGeoreference(const pxr::SdfPath& path)
    : _path(path) {}

pxr::SdfPath OmniGeoreference::getPath() const {
    return _path;
}

CesiumGeospatial::Cartographic OmniGeoreference::getOrigin() const {
    auto georeference = UsdUtil::getCesiumGeoreference(_path);

    double longitude;
    double latitude;
    double height;

    georeference.GetGeoreferenceOriginLongitudeAttr().Get<double>(&longitude);
    georeference.GetGeoreferenceOriginLatitudeAttr().Get<double>(&latitude);
    georeference.GetGeoreferenceOriginHeightAttr().Get<double>(&height);

    return {glm::radians(longitude), glm::radians(latitude), height};
}

void OmniGeoreference::setOrigin(const CesiumGeospatial::Cartographic& origin) const {
    auto georeference = UsdUtil::getCesiumGeoreference(_path);

    georeference.GetGeoreferenceOriginLongitudeAttr().Set<double>(glm::degrees(origin.longitude));
    georeference.GetGeoreferenceOriginLatitudeAttr().Set<double>(glm::degrees(origin.latitude));
    georeference.GetGeoreferenceOriginHeightAttr().Set<double>(origin.height);
}

} // namespace cesium::omniverse
