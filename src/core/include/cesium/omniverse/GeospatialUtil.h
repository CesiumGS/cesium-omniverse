#pragma once

#include <CesiumGeospatial/Cartographic.h>
#include <CesiumGeospatial/LocalHorizontalCoordinateSystem.h>
#include <CesiumUsdSchemas/georeference.h>
#include <glm/glm.hpp>

namespace CesiumGeospatial {
class Cartographic;
}

namespace cesium::omniverse::GeospatialUtil {

CesiumGeospatial::Cartographic convertGeoreferenceToCartographic(const pxr::CesiumGeoreference& georeference);
[[maybe_unused]] CesiumGeospatial::LocalHorizontalCoordinateSystem
getCoordinateSystem(const pxr::CesiumGeoreference& georeference, const double scaleInMeters);

}; // namespace cesium::omniverse::GeospatialUtil
