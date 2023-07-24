#pragma once

#include <CesiumGeospatial/Cartographic.h>
#include <CesiumGeospatial/LocalHorizontalCoordinateSystem.h>
#include <CesiumUsdSchemas/georeference.h>
#include <glm/glm.hpp>

namespace CesiumGeospatial {
class Cartographic;
class LocalHorizontalCoordinateSystem;
} // namespace CesiumGeospatial

namespace cesium::omniverse::GeospatialUtil {

CesiumGeospatial::Cartographic convertGeoreferenceToCartographic(const pxr::CesiumGeoreference& georeference);

CesiumGeospatial::LocalHorizontalCoordinateSystem
getCoordinateSystem(const CesiumGeospatial::Cartographic& origin, const double scaleInMeters);

}; // namespace cesium::omniverse::GeospatialUtil
