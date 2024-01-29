#include "cesium/omniverse/OmniCartographicPolygon.h"

#include "cesium/omniverse/AssetRegistry.h"
#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/OmniGeoreference.h"
#include "cesium/omniverse/OmniGlobeAnchor.h"
#include "cesium/omniverse/UsdUtil.h"

#include <CesiumGeospatial/Cartographic.h>
#include <CesiumUsdSchemas/cartographicPolygon.h>
#include <CesiumUsdSchemas/globeAnchorAPI.h>
#include <glm/glm.hpp>

namespace cesium::omniverse {

OmniCartographicPolygon::OmniCartographicPolygon(Context* pContext, const pxr::SdfPath& path)
    : _pContext(pContext)
    , _path(path) {}

const pxr::SdfPath& OmniCartographicPolygon::getPath() const {
    return _path;
}

std::vector<CesiumGeospatial::Cartographic> OmniCartographicPolygon::getCartographics() const {
    const auto pGlobeAnchor = _pContext->getAssetRegistry().getGlobeAnchor(_path);
    if (!pGlobeAnchor) {
        return {};
    }

    const auto georeferencePath = pGlobeAnchor->getResolvedGeoreferencePath();
    if (georeferencePath.IsEmpty()) {
        return {};
    }

    const auto pGeoreference = _pContext->getAssetRegistry().getGeoreference(georeferencePath);
    if (!pGeoreference) {
        return {};
    }

    const auto cesiumCartographicPolygon = UsdUtil::getCesiumCartographicPolygon(_pContext->getUsdStage(), _path);

    pxr::VtArray<pxr::GfVec3f> points;
    cesiumCartographicPolygon.GetPointsAttr().Get(&points);

    std::vector<glm::dvec3> positionsLocal;
    positionsLocal.reserve(points.size());
    for (const auto& point : points) {
        positionsLocal.push_back(glm::dvec3(UsdUtil::usdToGlmVector(point)));
    }

    const auto primLocalToEcefTransform = UsdUtil::computePrimLocalToEcefTransform(*_pContext, georeferencePath, _path);

    std::vector<CesiumGeospatial::Cartographic> cartographics;
    cartographics.reserve(positionsLocal.size());

    for (const auto& positionLocal : positionsLocal) {
        const auto positionEcef = glm::dvec3(primLocalToEcefTransform * glm::dvec4(positionLocal, 1.0));
        const auto positionCartographic = pGeoreference->getEllipsoid().cartesianToCartographic(positionEcef);

        if (!positionCartographic.has_value()) {
            return {};
        }

        cartographics.push_back(positionCartographic.value());
    }

    return cartographics;
}

} // namespace cesium::omniverse
