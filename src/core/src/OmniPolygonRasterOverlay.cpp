#include "cesium/omniverse/OmniPolygonRasterOverlay.h"

#include "cesium/omniverse/AssetRegistry.h"
#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/GltfUtil.h"
#include "cesium/omniverse/Logger.h"
#include "cesium/omniverse/OmniCartographicPolygon.h"
#include "cesium/omniverse/OmniGeoreference.h"
#include "cesium/omniverse/OmniGlobeAnchor.h"
#include "cesium/omniverse/UsdUtil.h"

#include <CesiumGeospatial/Ellipsoid.h>
#include <CesiumRasterOverlays/RasterizedPolygonsOverlay.h>
#include <CesiumUsdSchemas/polygonRasterOverlay.h>

#include <memory>

namespace cesium::omniverse {

OmniPolygonRasterOverlay::OmniPolygonRasterOverlay(Context* pContext, const pxr::SdfPath& path)
    : OmniRasterOverlay(pContext, path)
    , _rasterizedPolygonsTileExcluder(nullptr) {}

std::vector<pxr::SdfPath> OmniPolygonRasterOverlay::getCartographicPolygonPaths() const {
    const auto cesiumPolygonRasterOverlay = UsdUtil::getCesiumPolygonRasterOverlay(_pContext->getUsdStage(), _path);

    pxr::SdfPathVector targets;
    cesiumPolygonRasterOverlay.GetCartographicPolygonBindingRel().GetForwardedTargets(&targets);

    return targets;
}

CesiumRasterOverlays::RasterOverlay* OmniPolygonRasterOverlay::getRasterOverlay() const {
    return _pPolygonRasterOverlay.get();
}

bool OmniPolygonRasterOverlay::getInvertSelection() const {
    auto cesiumPolygonRasterOverlay = UsdUtil::getCesiumPolygonRasterOverlay(_pContext->getUsdStage(), _path);
    bool invertSelection;
    cesiumPolygonRasterOverlay.GetInvertSelectionAttr().Get(&invertSelection);
    return invertSelection;
}

bool OmniPolygonRasterOverlay::getExcludeSelectedTiles() const {
    auto cesiumPolygonRasterOverlay = UsdUtil::getCesiumPolygonRasterOverlay(_pContext->getUsdStage(), _path);
    bool excludeSelectedTiles;
    cesiumPolygonRasterOverlay.GetExcludeSelectedTilesAttr().Get(&excludeSelectedTiles);
    return excludeSelectedTiles;
}

void OmniPolygonRasterOverlay::reload() {
    const auto rasterOverlayName = UsdUtil::getName(_pContext->getUsdStage(), _path);

    const auto cartographicPolygonPaths = getCartographicPolygonPaths();
    std::vector<CesiumGeospatial::CartographicPolygon> polygons;

    const CesiumGeospatial::Ellipsoid* pEllipsoid = nullptr;
    _rasterizedPolygonsTileExcluder = nullptr;

    for (const auto& cartographicPolygonPath : cartographicPolygonPaths) {
        const auto pCartographicPolygon = _pContext->getAssetRegistry().getCartographicPolygon(cartographicPolygonPath);
        if (!pCartographicPolygon) {
            continue;
        }

        const auto pGlobeAnchor = _pContext->getAssetRegistry().getGlobeAnchor(cartographicPolygonPath);
        if (!pGlobeAnchor) {
            continue;
        }

        const auto georeferencePath = pGlobeAnchor->getResolvedGeoreferencePath();
        if (georeferencePath.IsEmpty()) {
            continue;
        }

        const auto pGeoreference = _pContext->getAssetRegistry().getGeoreference(georeferencePath);
        if (!pGeoreference) {
            continue;
        }

        const auto& ellipsoid = pGeoreference->getEllipsoid();

        if (!pEllipsoid) {
            pEllipsoid = &ellipsoid;
        } else if (*pEllipsoid != ellipsoid) {
            return; // All cartographic polygons must use the same ellipsoid
        }

        const auto cartographics = pCartographicPolygon->getCartographics();

        std::vector<glm::dvec2> polygon;
        for (const auto& cartographic : cartographics) {
            polygon.emplace_back(cartographic.longitude, cartographic.latitude);
        }
        polygons.emplace_back(polygon);
    }

    if (polygons.empty()) {
        return;
    }

    if (!pEllipsoid) {
        return;
    }

    if (getExcludeSelectedTiles()) {
        _rasterizedPolygonsTileExcluder =
            std::make_shared<Cesium3DTilesSelection::RasterizedPolygonsTileExcluder>(_pPolygonRasterOverlay);
        // pTileset->getOptions().excluders.push_back(this->_pExcluder);
    }

    const auto projection = CesiumGeospatial::GeographicProjection(*pEllipsoid);

    CesiumRasterOverlays::RasterOverlayOptions options;
    options.showCreditsOnScreen = getShowCreditsOnScreen();
    options.ktx2TranscodeTargets = GltfUtil::getKtx2TranscodeTargets();

    options.loadErrorCallback = [this](const CesiumRasterOverlays::RasterOverlayLoadFailureDetails& error) {
        _pContext->getLogger()->error(error.message);
    };

    _pPolygonRasterOverlay = new CesiumRasterOverlays::RasterizedPolygonsOverlay(
        rasterOverlayName, polygons, getInvertSelection(), *pEllipsoid, projection, options);
}

} // namespace cesium::omniverse
