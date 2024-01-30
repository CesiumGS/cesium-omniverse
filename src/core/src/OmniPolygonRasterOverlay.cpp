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

namespace cesium::omniverse {

OmniPolygonRasterOverlay::OmniPolygonRasterOverlay(Context* pContext, const pxr::SdfPath& path)
    : OmniRasterOverlay(pContext, path) {}

std::vector<pxr::SdfPath> OmniPolygonRasterOverlay::getCartographicPolygonPaths() const {
    const auto cesiumPolygonRasterOverlay = UsdUtil::getCesiumPolygonRasterOverlay(_pContext->getUsdStage(), _path);

    pxr::SdfPathVector targets;
    cesiumPolygonRasterOverlay.GetCartographicPolygonBindingRel().GetForwardedTargets(&targets);

    return targets;
}

CesiumRasterOverlays::RasterOverlay* OmniPolygonRasterOverlay::getRasterOverlay() const {
    return _pPolygonRasterOverlay.get();
}

void OmniPolygonRasterOverlay::reload() {
    const auto rasterOverlayName = UsdUtil::getName(_pContext->getUsdStage(), _path);

    const auto cartographicPolygonPaths = getCartographicPolygonPaths();
    std::vector<CesiumGeospatial::CartographicPolygon> polygons;

    const CesiumGeospatial::Ellipsoid* pEllipsoid = nullptr;

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
        polygons.push_back(polygon);
    }

    if (polygons.empty()) {
        return;
    }

    if (!pEllipsoid) {
        return;
    }

    auto invertSelection = false;
    auto& path = getPath();
    auto& usdStage = _pContext->getUsdStage();
    const auto prim = usdStage->GetPrimAtPath(path);
    if (prim.IsValid()) {
        const auto cesiumPolygonRasterOverlay = UsdUtil::getCesiumPolygonRasterOverlay(_pContext->getUsdStage(), _path);
        auto attr = cesiumPolygonRasterOverlay.GetInvertSelectionAttr();
        if (attr.IsValid()) {
            bool value;
            attr.Get(&value);
            invertSelection = value;
        }
    }

    const auto projection = CesiumGeospatial::GeographicProjection(*pEllipsoid);

    CesiumRasterOverlays::RasterOverlayOptions options;
    options.showCreditsOnScreen = getShowCreditsOnScreen();
    options.ktx2TranscodeTargets = GltfUtil::getKtx2TranscodeTargets();

    options.loadErrorCallback = [this](const CesiumRasterOverlays::RasterOverlayLoadFailureDetails& error) {
        _pContext->getLogger()->error(error.message);
    };

    _pPolygonRasterOverlay = new CesiumRasterOverlays::RasterizedPolygonsOverlay(
        rasterOverlayName, polygons, invertSelection, *pEllipsoid, projection, options);
}

} // namespace cesium::omniverse
