#include "cesium/omniverse/OmniTileMapServiceRasterOverlay.h"

#include "cesium/omniverse/AssetRegistry.h"
#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/GltfUtil.h"
#include "cesium/omniverse/Logger.h"
#include "cesium/omniverse/OmniGeoreference.h"
#include "cesium/omniverse/UsdUtil.h"

#include <CesiumRasterOverlays/TileMapServiceRasterOverlay.h>
#include <CesiumUsdSchemas/tileMapServiceRasterOverlay.h>

namespace cesium::omniverse {

OmniTileMapServiceRasterOverlay::OmniTileMapServiceRasterOverlay(Context* pContext, const pxr::SdfPath& path)
    : OmniRasterOverlay(pContext, path) {
    reload();
}

CesiumRasterOverlays::RasterOverlay* OmniTileMapServiceRasterOverlay::getRasterOverlay() const {
    return _pTileMapServiceRasterOverlay.get();
}

std::string OmniTileMapServiceRasterOverlay::getUrl() const {
    const auto cesiumTileMapServiceRasterOverlay =
        UsdUtil::getCesiumTileMapServiceRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumTileMapServiceRasterOverlay)) {
        return "";
    }

    std::string url;
    cesiumTileMapServiceRasterOverlay.GetUrlAttr().Get(&url);
    return url;
}

int OmniTileMapServiceRasterOverlay::getMinimumZoomLevel() const {
    const auto cesiumTileMapServiceRasterOverlay =
        UsdUtil::getCesiumTileMapServiceRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumTileMapServiceRasterOverlay)) {
        return 0;
    }

    int minimumZoomLevel;
    cesiumTileMapServiceRasterOverlay.GetMinimumZoomLevelAttr().Get(&minimumZoomLevel);
    return minimumZoomLevel;
}

int OmniTileMapServiceRasterOverlay::getMaximumZoomLevel() const {
    const auto cesiumTileMapServiceRasterOverlay =
        UsdUtil::getCesiumTileMapServiceRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumTileMapServiceRasterOverlay)) {
        return 10;
    }

    int maximumZoomLevel;
    cesiumTileMapServiceRasterOverlay.GetMaximumZoomLevelAttr().Get(&maximumZoomLevel);
    return maximumZoomLevel;
}

bool OmniTileMapServiceRasterOverlay::getSpecifyZoomLevels() const {
    const auto cesiumTileMapServiceRasterOverlay =
        UsdUtil::getCesiumTileMapServiceRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumTileMapServiceRasterOverlay)) {
        return false;
    }

    bool value;
    cesiumTileMapServiceRasterOverlay.GetSpecifyZoomLevelsAttr().Get(&value);
    return value;
}

void OmniTileMapServiceRasterOverlay::reload() {
    const auto rasterOverlayName = UsdUtil::getName(_pContext->getUsdStage(), _path);

    auto options = createRasterOverlayOptions();

    options.loadErrorCallback = [this](const CesiumRasterOverlays::RasterOverlayLoadFailureDetails& error) {
        _pContext->getLogger()->error(error.message);
    };

    CesiumRasterOverlays::TileMapServiceRasterOverlayOptions tmsOptions;
    const auto specifyZoomLevels = getSpecifyZoomLevels();

    if (specifyZoomLevels) {
        tmsOptions.minimumLevel = getMinimumZoomLevel();
        tmsOptions.maximumLevel = getMaximumZoomLevel();
    }

    tmsOptions.ellipsoid = getEllipsoid();

    _pTileMapServiceRasterOverlay = new CesiumRasterOverlays::TileMapServiceRasterOverlay(
        rasterOverlayName, getUrl(), std::vector<CesiumAsync::IAssetAccessor::THeader>(), tmsOptions, options);
}

} // namespace cesium::omniverse
