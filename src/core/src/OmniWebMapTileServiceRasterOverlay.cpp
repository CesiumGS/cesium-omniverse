#include "cesium/omniverse/OmniWebMapTileServiceRasterOverlay.h"

#include "cesium/omniverse/AssetRegistry.h"
#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/GltfUtil.h"
#include "cesium/omniverse/Logger.h"
#include "cesium/omniverse/OmniGeoreference.h"
#include "cesium/omniverse/UsdUtil.h"

#include <CesiumRasterOverlays/WebMapTileServiceRasterOverlay.h>
#include <CesiumUsdSchemas/webMapTileServiceRasterOverlay.h>

#include <string>

namespace cesium::omniverse {

OmniWebMapTileServiceRasterOverlay::OmniWebMapTileServiceRasterOverlay(Context* pContext, const pxr::SdfPath& path)
    : OmniRasterOverlay(pContext, path) {
    reload();
}

CesiumRasterOverlays::RasterOverlay* OmniWebMapTileServiceRasterOverlay::getRasterOverlay() const {
    return _pWebMapTileServiceRasterOverlay.get();
}

std::string OmniWebMapTileServiceRasterOverlay::getUrl() const {
    const auto cesiumWebMapTileServiceRasterOverlay =
        UsdUtil::getCesiumWebMapTileServiceRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumWebMapTileServiceRasterOverlay)) {
        return "";
    }

    std::string url;
    cesiumWebMapTileServiceRasterOverlay.GetUrlAttr().Get(&url);
    return url;
}

std::string OmniWebMapTileServiceRasterOverlay::getLayer() const {
    const auto cesiumWebMapTileServiceRasterOverlay =
        UsdUtil::getCesiumWebMapTileServiceRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumWebMapTileServiceRasterOverlay)) {
        return "";
    }

    std::string layer;
    cesiumWebMapTileServiceRasterOverlay.GetLayerAttr().Get(&layer);
    return layer;
}

std::string OmniWebMapTileServiceRasterOverlay::getTileMatrixSetId() const {
    const auto cesiumWebMapTileServiceRasterOverlay =
        UsdUtil::getCesiumWebMapTileServiceRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumWebMapTileServiceRasterOverlay)) {
        return "";
    }

    std::string val;
    cesiumWebMapTileServiceRasterOverlay.GetTileMatrixSetIdAttr().Get(&val);
    return val;
}

std::string OmniWebMapTileServiceRasterOverlay::getStyle() const {
    const auto cesiumWebMapTileServiceRasterOverlay =
        UsdUtil::getCesiumWebMapTileServiceRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumWebMapTileServiceRasterOverlay)) {
        return "";
    }

    std::string val;
    cesiumWebMapTileServiceRasterOverlay.GetStyleAttr().Get(&val);
    return val;
}

std::string OmniWebMapTileServiceRasterOverlay::getFormat() const {
    const auto cesiumWebMapTileServiceRasterOverlay =
        UsdUtil::getCesiumWebMapTileServiceRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumWebMapTileServiceRasterOverlay)) {
        return "";
    }

    std::string val;
    cesiumWebMapTileServiceRasterOverlay.GetFormatAttr().Get(&val);
    return val;
}

int OmniWebMapTileServiceRasterOverlay::getMinimumZoomLevel() const {
    const auto cesiumWebMapTileServiceRasterOverlay =
        UsdUtil::getCesiumWebMapTileServiceRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumWebMapTileServiceRasterOverlay)) {
        return 0;
    }

    int val;
    cesiumWebMapTileServiceRasterOverlay.GetMinimumZoomLevelAttr().Get(&val);
    return val;
}

int OmniWebMapTileServiceRasterOverlay::getMaximumZoomLevel() const {
    const auto cesiumWebMapTileServiceRasterOverlay =
        UsdUtil::getCesiumWebMapTileServiceRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumWebMapTileServiceRasterOverlay)) {
        return 0;
    }

    int val;
    cesiumWebMapTileServiceRasterOverlay.GetMaximumZoomLevelAttr().Get(&val);
    return val;
}

bool OmniWebMapTileServiceRasterOverlay::getSpecifyZoomLevels() const {
    const auto cesiumWebMapTileServiceRasterOverlay =
        UsdUtil::getCesiumWebMapTileServiceRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumWebMapTileServiceRasterOverlay)) {
        return false;
    }

    bool val;
    cesiumWebMapTileServiceRasterOverlay.GetSpecifyZoomLevelsAttr().Get(&val);
    return val;
}

bool OmniWebMapTileServiceRasterOverlay::getUseWebMercatorProjection() const {
    const auto cesiumWebMapTileServiceRasterOverlay =
        UsdUtil::getCesiumWebMapTileServiceRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumWebMapTileServiceRasterOverlay)) {
        return false;
    }

    bool val;
    cesiumWebMapTileServiceRasterOverlay.GetUseWebMercatorProjectionAttr().Get(&val);
    return val;
}

bool OmniWebMapTileServiceRasterOverlay::getSpecifyTilingScheme() const {
    const auto cesiumWebMapTileServiceRasterOverlay =
        UsdUtil::getCesiumWebMapTileServiceRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumWebMapTileServiceRasterOverlay)) {
        return false;
    }

    bool val;
    cesiumWebMapTileServiceRasterOverlay.GetSpecifyTilingSchemeAttr().Get(&val);
    return val;
}

double OmniWebMapTileServiceRasterOverlay::getNorth() const {
    const auto cesiumWebMapTileServiceRasterOverlay =
        UsdUtil::getCesiumWebMapTileServiceRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumWebMapTileServiceRasterOverlay)) {
        return 90;
    }

    double val;
    cesiumWebMapTileServiceRasterOverlay.GetNorthAttr().Get(&val);
    return val;
}

double OmniWebMapTileServiceRasterOverlay::getSouth() const {
    const auto cesiumWebMapTileServiceRasterOverlay =
        UsdUtil::getCesiumWebMapTileServiceRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumWebMapTileServiceRasterOverlay)) {
        return -90;
    }

    double val;
    cesiumWebMapTileServiceRasterOverlay.GetSouthAttr().Get(&val);
    return val;
}

double OmniWebMapTileServiceRasterOverlay::getEast() const {
    const auto cesiumWebMapTileServiceRasterOverlay =
        UsdUtil::getCesiumWebMapTileServiceRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumWebMapTileServiceRasterOverlay)) {
        return 180;
    }

    double val;
    cesiumWebMapTileServiceRasterOverlay.GetEastAttr().Get(&val);
    return val;
}

double OmniWebMapTileServiceRasterOverlay::getWest() const {
    const auto cesiumWebMapTileServiceRasterOverlay =
        UsdUtil::getCesiumWebMapTileServiceRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumWebMapTileServiceRasterOverlay)) {
        return -180;
    }

    double val;
    cesiumWebMapTileServiceRasterOverlay.GetWestAttr().Get(&val);
    return val;
}

bool OmniWebMapTileServiceRasterOverlay::getSpecifyTileMatrixSetLabels() const {
    const auto cesiumWebMapTileServiceRasterOverlay =
        UsdUtil::getCesiumWebMapTileServiceRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumWebMapTileServiceRasterOverlay)) {
        return false;
    }

    bool val;
    cesiumWebMapTileServiceRasterOverlay.GetSpecifyTileMatrixSetLabelsAttr().Get(&val);
    return val;
}

std::vector<std::string> OmniWebMapTileServiceRasterOverlay::getTileMatrixSetLabels() const {
    const auto cesiumWebMapTileServiceRasterOverlay =
        UsdUtil::getCesiumWebMapTileServiceRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumWebMapTileServiceRasterOverlay)) {
        return {};
    }

    std::string val;
    cesiumWebMapTileServiceRasterOverlay.GetTileMatrixSetLabelsAttr().Get(&val);
    std::vector<std::string> matrixSetLabels;

    size_t pos = 0;
    while ((pos = val.find(',')) != std::string::npos) {
        matrixSetLabels.push_back(val.substr(0, pos));
        val.erase(0, pos + 1);
    }
    matrixSetLabels.push_back(val);
    return matrixSetLabels;
}

std::string OmniWebMapTileServiceRasterOverlay::getTileMatrixSetLabelPrefix() const {
    const auto cesiumWebMapTileServiceRasterOverlay =
        UsdUtil::getCesiumWebMapTileServiceRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumWebMapTileServiceRasterOverlay)) {
        return "";
    }

    std::string val;
    cesiumWebMapTileServiceRasterOverlay.GetTileMatrixSetLabelPrefixAttr().Get(&val);
    return val;
}

int OmniWebMapTileServiceRasterOverlay::getRootTilesX() const {
    const auto cesiumWebMapTileServiceRasterOverlay =
        UsdUtil::getCesiumWebMapTileServiceRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumWebMapTileServiceRasterOverlay)) {
        return 1;
    }

    int val;
    cesiumWebMapTileServiceRasterOverlay.GetRootTilesXAttr().Get(&val);
    return val;
}

int OmniWebMapTileServiceRasterOverlay::getRootTilesY() const {
    const auto cesiumWebMapTileServiceRasterOverlay =
        UsdUtil::getCesiumWebMapTileServiceRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumWebMapTileServiceRasterOverlay)) {
        return 1;
    }

    int val;
    cesiumWebMapTileServiceRasterOverlay.GetRootTilesYAttr().Get(&val);
    return val;
}

void OmniWebMapTileServiceRasterOverlay::reload() {
    const auto rasterOverlayName = UsdUtil::getName(_pContext->getUsdStage(), _path);

    auto options = createRasterOverlayOptions();

    options.loadErrorCallback = [this](const CesiumRasterOverlays::RasterOverlayLoadFailureDetails& error) {
        _pContext->getLogger()->error(error.message);
    };

    CesiumRasterOverlays::WebMapTileServiceRasterOverlayOptions wmtsOptions;

    const auto tileMatrixSetId = getTileMatrixSetId();
    if (!tileMatrixSetId.empty()) {
        wmtsOptions.tileMatrixSetID = tileMatrixSetId;
    }

    const auto style = getStyle();
    if (!style.empty()) {
        wmtsOptions.style = style;
    }

    const auto layer = getLayer();
    if (!layer.empty()) {
        wmtsOptions.layer = layer;
    }

    const auto format = getFormat();
    if (!format.empty()) {
        wmtsOptions.format = format;
    }

    if (getSpecifyZoomLevels()) {
        wmtsOptions.minimumLevel = getMinimumZoomLevel();
        wmtsOptions.maximumLevel = getMaximumZoomLevel();
    }

    const auto& ellipsoid = getEllipsoid();

    wmtsOptions.ellipsoid = ellipsoid;

    const auto useWebMercatorProjection = getUseWebMercatorProjection();
    if (useWebMercatorProjection) {
        wmtsOptions.projection = CesiumGeospatial::WebMercatorProjection(ellipsoid);
    } else {
        wmtsOptions.projection = CesiumGeospatial::GeographicProjection(ellipsoid);
    }

    if (getSpecifyTilingScheme()) {
        CesiumGeospatial::GlobeRectangle globeRectangle =
            CesiumGeospatial::GlobeRectangle::fromDegrees(getWest(), getSouth(), getEast(), getNorth());
        CesiumGeometry::Rectangle coverageRectangle =
            CesiumGeospatial::projectRectangleSimple(wmtsOptions.projection.value(), globeRectangle);
        wmtsOptions.coverageRectangle = coverageRectangle;
        const auto rootTilesX = getRootTilesX();
        const auto rootTilesY = getRootTilesY();
        wmtsOptions.tilingScheme = CesiumGeometry::QuadtreeTilingScheme(coverageRectangle, rootTilesX, rootTilesY);
    }

    if (getSpecifyTileMatrixSetLabels()) {
        const auto tileMatrixSetLabels = getTileMatrixSetLabels();
        if (!tileMatrixSetLabels.empty()) {
            wmtsOptions.tileMatrixLabels = getTileMatrixSetLabels();
        }
    } else {
        const auto tileMatrixSetLabelPrefix = getTileMatrixSetLabelPrefix();
        if (!tileMatrixSetLabelPrefix.empty()) {
            std::vector<std::string> labels;
            for (size_t level = 0; level <= 25; ++level) {
                std::string label{tileMatrixSetLabelPrefix};
                label.append(std::to_string(level));
                labels.emplace_back(label);
            }
            wmtsOptions.tileMatrixLabels = labels;
        }
    }

    _pWebMapTileServiceRasterOverlay = new CesiumRasterOverlays::WebMapTileServiceRasterOverlay(
        rasterOverlayName, getUrl(), std::vector<CesiumAsync::IAssetAccessor::THeader>(), wmtsOptions, options);
}

} // namespace cesium::omniverse
