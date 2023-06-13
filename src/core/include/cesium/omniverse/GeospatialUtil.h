#pragma once

#include <CesiumGeospatial/Cartographic.h>
#include <CesiumUsdSchemas/georeference.h>
#include <glm/glm.hpp>

namespace CesiumGeospatial {
class Cartographic;
}

namespace cesium::omniverse::GeospatialUtil {

CesiumGeospatial::Cartographic convertGeoreferenceToCartographic(const pxr::CesiumGeoreference& georeference);

}; // namespace cesium::omniverse::GeospatialUtil
