#pragma once

#include "CesiumUsdSchemas/globeAnchorAPI.h"

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

glm::dmat4 getAxisConversionTransform();
glm::dmat4 getEastNorthUpToFixedFrame(const CesiumGeospatial::Cartographic& cartographic);
[[maybe_unused]] CesiumGeospatial::LocalHorizontalCoordinateSystem
getCoordinateSystem(const pxr::CesiumGeoreference& georeference, const double scaleInMeters);
CesiumGeospatial::LocalHorizontalCoordinateSystem
getCoordinateSystem(const CesiumGeospatial::Cartographic& origin, const double scaleInMeters);
glm::dmat4 getUnitConversionTransform();

void updateAnchorByUsdTransform(const CesiumGeospatial::Cartographic& origin, const pxr::CesiumGlobeAnchorAPI& anchor);
void updateAnchorByLatLongHeight(const CesiumGeospatial::Cartographic& origin, const pxr::CesiumGlobeAnchorAPI& anchor);
void updateAnchorByFixedTransform(
    const CesiumGeospatial::Cartographic& origin,
    const pxr::CesiumGlobeAnchorAPI& anchor);

}; // namespace cesium::omniverse::GeospatialUtil
