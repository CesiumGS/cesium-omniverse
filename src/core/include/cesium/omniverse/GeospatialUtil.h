#pragma once

#include <CesiumGeospatial/Cartographic.h>
#include <CesiumGeospatial/LocalHorizontalCoordinateSystem.h>
#include <CesiumUsdSchemas/georeference.h>
#include <CesiumUsdSchemas/globeAnchorAPI.h>
#include <glm/glm.hpp>

namespace CesiumGeospatial {
class Cartographic;
class LocalHorizontalCoordinateSystem;
} // namespace CesiumGeospatial

namespace cesium::omniverse {

class OmniGlobeAnchor;

namespace GeospatialUtil {

CesiumGeospatial::Cartographic convertGeoreferenceToCartographic(const pxr::CesiumGeoreference& georeference);

CesiumGeospatial::LocalHorizontalCoordinateSystem getCoordinateSystem(const CesiumGeospatial::Cartographic& origin);

void updateAnchorByUsdTransform(
    const CesiumGeospatial::Cartographic& origin,
    const pxr::CesiumGlobeAnchorAPI& anchorApi);
void updateAnchorByLatLongHeight(
    const CesiumGeospatial::Cartographic& origin,
    const pxr::CesiumGlobeAnchorAPI& anchorApi);
void updateAnchorByFixedTransform(
    const CesiumGeospatial::Cartographic& origin,
    const pxr::CesiumGlobeAnchorAPI& anchorApi);
void updateAnchorOrigin(
    const CesiumGeospatial::Cartographic& origin,
    const pxr::CesiumGlobeAnchorAPI& anchorApi,
    const std::shared_ptr<OmniGlobeAnchor>& globeAnchor);

}; // namespace GeospatialUtil
}; // namespace cesium::omniverse
