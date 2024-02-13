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
    auto cesiumWebMapServiceRasterOverlay =
        UsdUtil::getCesiumWebMapServiceRasterOverlay(_pContext->getUsdStage(), _path);
    std::string baseUrl;
    cesiumWebMapServiceRasterOverlay.GetBaseUrlAttr().Get(&baseUrl);
    return baseUrl;
}

int OmniWebMapServiceRasterOverlay::getMinimumLevel() const {
    auto cesiumWebMapServiceRasterOverlay =
        UsdUtil::getCesiumWebMapServiceRasterOverlay(_pContext->getUsdStage(), _path);
    int val;
    cesiumWebMapServiceRasterOverlay.GetMinimumLevelAttr().Get(&val);
    return val;
}

int OmniWebMapServiceRasterOverlay::getMaximumLevel() const {
    auto cesiumWebMapServiceRasterOverlay =
        UsdUtil::getCesiumWebMapServiceRasterOverlay(_pContext->getUsdStage(), _path);
    int val;
    cesiumWebMapServiceRasterOverlay.GetMaximumLevelAttr().Get(&val);
    return val;
}

int OmniWebMapServiceRasterOverlay::getTileWidth() const {
    auto cesiumWebMapServiceRasterOverlay =
        UsdUtil::getCesiumWebMapServiceRasterOverlay(_pContext->getUsdStage(), _path);
    int val;
    cesiumWebMapServiceRasterOverlay.GetTileWidthAttr().Get(&val);
    return val;
}

int OmniWebMapServiceRasterOverlay::getTileHeight() const {
    auto cesiumWebMapServiceRasterOverlay =
        UsdUtil::getCesiumWebMapServiceRasterOverlay(_pContext->getUsdStage(), _path);
    int val;
    cesiumWebMapServiceRasterOverlay.GetTileHeightAttr().Get(&val);
    return val;
}

std::string OmniWebMapServiceRasterOverlay::getLayers() const {
    auto cesiumWebMapServiceRasterOverlay =
        UsdUtil::getCesiumWebMapServiceRasterOverlay(_pContext->getUsdStage(), _path);
    std::string val;
    cesiumWebMapServiceRasterOverlay.GetLayersAttr().Get(&val);
    return val;
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
