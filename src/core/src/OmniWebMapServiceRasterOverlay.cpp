#include "cesium/omniverse/OmniWebMapServiceRasterOverlay.h"

#include "cesium/omniverse/AssetRegistry.h"
#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/GltfUtil.h"
#include "cesium/omniverse/Logger.h"
#include "cesium/omniverse/OmniGeoreference.h"
#include "cesium/omniverse/UsdUtil.h"

#include <CesiumRasterOverlays/WebMapServiceRasterOverlay.h>
#include <CesiumUsdSchemas/webMapServiceRasterOverlay.h>

namespace cesium::omniverse {

OmniWebMapServiceRasterOverlay::OmniWebMapServiceRasterOverlay(Context* pContext, const pxr::SdfPath& path)
    : OmniRasterOverlay(pContext, path) {
    reload();
}

CesiumRasterOverlays::RasterOverlay* OmniWebMapServiceRasterOverlay::getRasterOverlay() const {
    return _pWebMapServiceRasterOverlay.get();
}

std::string OmniWebMapServiceRasterOverlay::getBaseUrl() const {
    const auto cesiumWebMapServiceRasterOverlay =
        UsdUtil::getCesiumWebMapServiceRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumWebMapServiceRasterOverlay)) {
        return "";
    }

    std::string baseUrl;
    cesiumWebMapServiceRasterOverlay.GetBaseUrlAttr().Get(&baseUrl);
    return baseUrl;
}

int OmniWebMapServiceRasterOverlay::getMinimumLevel() const {
    const auto cesiumWebMapServiceRasterOverlay =
        UsdUtil::getCesiumWebMapServiceRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumWebMapServiceRasterOverlay)) {
        return 0;
    }

    int minimumLevel;
    cesiumWebMapServiceRasterOverlay.GetMinimumLevelAttr().Get(&minimumLevel);
    return minimumLevel;
}

int OmniWebMapServiceRasterOverlay::getMaximumLevel() const {
    const auto cesiumWebMapServiceRasterOverlay =
        UsdUtil::getCesiumWebMapServiceRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumWebMapServiceRasterOverlay)) {
        return 14;
    }

    int maximumLevel;
    cesiumWebMapServiceRasterOverlay.GetMaximumLevelAttr().Get(&maximumLevel);
    return maximumLevel;
}

int OmniWebMapServiceRasterOverlay::getTileWidth() const {
    const auto cesiumWebMapServiceRasterOverlay =
        UsdUtil::getCesiumWebMapServiceRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumWebMapServiceRasterOverlay)) {
        return 256;
    }

    int tileWidth;
    cesiumWebMapServiceRasterOverlay.GetTileWidthAttr().Get(&tileWidth);
    return tileWidth;
}

int OmniWebMapServiceRasterOverlay::getTileHeight() const {
    const auto cesiumWebMapServiceRasterOverlay =
        UsdUtil::getCesiumWebMapServiceRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumWebMapServiceRasterOverlay)) {
        return 256;
    }

    int tileHeight;
    cesiumWebMapServiceRasterOverlay.GetTileHeightAttr().Get(&tileHeight);
    return tileHeight;
}

std::string OmniWebMapServiceRasterOverlay::getLayers() const {
    const auto cesiumWebMapServiceRasterOverlay =
        UsdUtil::getCesiumWebMapServiceRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumWebMapServiceRasterOverlay)) {
        return "1";
    }

    std::string layers;
    cesiumWebMapServiceRasterOverlay.GetLayersAttr().Get(&layers);
    return layers;
}

void OmniWebMapServiceRasterOverlay::reload() {
    const auto rasterOverlayName = UsdUtil::getName(_pContext->getUsdStage(), _path);

    auto options = createRasterOverlayOptions();

    options.loadErrorCallback = [this](const CesiumRasterOverlays::RasterOverlayLoadFailureDetails& error) {
        _pContext->getLogger()->error(error.message);
    };

    CesiumRasterOverlays::WebMapServiceRasterOverlayOptions wmsOptions;
    const auto minimumLevel = getMinimumLevel();
    const auto maximumLevel = getMaximumLevel();
    const auto tileWidth = getTileWidth();
    const auto tileHeight = getTileHeight();
    std::string layers = getLayers();

    wmsOptions.minimumLevel = minimumLevel;
    wmsOptions.maximumLevel = maximumLevel;
    wmsOptions.layers = layers;
    wmsOptions.tileWidth = tileWidth;
    wmsOptions.tileHeight = tileHeight;

    _pWebMapServiceRasterOverlay = new CesiumRasterOverlays::WebMapServiceRasterOverlay(
        rasterOverlayName, getBaseUrl(), std::vector<CesiumAsync::IAssetAccessor::THeader>(), wmsOptions, options);
}

} // namespace cesium::omniverse
