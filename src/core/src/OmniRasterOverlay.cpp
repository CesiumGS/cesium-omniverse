#include "cesium/omniverse/OmniRasterOverlay.h"

#include "cesium/omniverse/AssetRegistry.h"
#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/FabricRasterOverlaysInfo.h"
#include "cesium/omniverse/GltfUtil.h"
#include "cesium/omniverse/Logger.h"
#include "cesium/omniverse/OmniIonServer.h"
#include "cesium/omniverse/UsdUtil.h"

#include <CesiumGeospatial/Ellipsoid.h>
#include <CesiumIonClient/Token.h>
#include <CesiumUsdSchemas/rasterOverlay.h>

namespace cesium::omniverse {

OmniRasterOverlay::OmniRasterOverlay(Context* pContext, const pxr::SdfPath& path)
    : _pContext(pContext)
    , _path(path) {}

const pxr::SdfPath& OmniRasterOverlay::getPath() const {
    return _path;
}

bool OmniRasterOverlay::getShowCreditsOnScreen() const {
    const auto cesiumRasterOverlay = UsdUtil::getCesiumRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumRasterOverlay)) {
        return false;
    }

    bool showCreditsOnScreen;
    cesiumRasterOverlay.GetShowCreditsOnScreenAttr().Get(&showCreditsOnScreen);

    return showCreditsOnScreen;
}

double OmniRasterOverlay::getAlpha() const {
    const auto cesiumRasterOverlay = UsdUtil::getCesiumRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumRasterOverlay)) {
        return 1.0;
    }

    float alpha;
    cesiumRasterOverlay.GetAlphaAttr().Get(&alpha);

    return static_cast<double>(alpha);
}

float OmniRasterOverlay::getMaximumScreenSpaceError() const {
    const auto cesiumRasterOverlay = UsdUtil::getCesiumRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumRasterOverlay)) {
        return 2.0f;
    }

    float value;
    cesiumRasterOverlay.GetMaximumScreenSpaceErrorAttr().Get(&value);

    return static_cast<float>(value);
}

int OmniRasterOverlay::getMaximumTextureSize() const {
    const auto cesiumRasterOverlay = UsdUtil::getCesiumRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumRasterOverlay)) {
        return 2048;
    }

    int value;
    cesiumRasterOverlay.GetMaximumTextureSizeAttr().Get(&value);

    return static_cast<int>(value);
}

int OmniRasterOverlay::getMaximumSimultaneousTileLoads() const {
    const auto cesiumRasterOverlay = UsdUtil::getCesiumRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumRasterOverlay)) {
        return 20;
    }

    int value;
    cesiumRasterOverlay.GetMaximumSimultaneousTileLoadsAttr().Get(&value);

    return static_cast<int>(value);
}

int OmniRasterOverlay::getSubTileCacheBytes() const {
    const auto cesiumRasterOverlay = UsdUtil::getCesiumRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumRasterOverlay)) {
        return 16777216;
    }

    int value;
    cesiumRasterOverlay.GetSubTileCacheBytesAttr().Get(&value);

    return static_cast<int>(value);
}

FabricOverlayRenderMethod OmniRasterOverlay::getOverlayRenderMethod() const {
    const auto cesiumRasterOverlay = UsdUtil::getCesiumRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumRasterOverlay)) {
        return FabricOverlayRenderMethod::OVERLAY;
    }

    pxr::TfToken overlayRenderMethod;
    cesiumRasterOverlay.GetOverlayRenderMethodAttr().Get(&overlayRenderMethod);

    if (overlayRenderMethod == pxr::CesiumTokens->overlay) {
        return FabricOverlayRenderMethod::OVERLAY;
    } else if (overlayRenderMethod == pxr::CesiumTokens->clip) {
        return FabricOverlayRenderMethod::CLIPPING;
    }

    _pContext->getLogger()->warn("Invalid overlay render method encountered {}.", overlayRenderMethod.GetText());
    return FabricOverlayRenderMethod::OVERLAY;
}

CesiumRasterOverlays::RasterOverlayOptions OmniRasterOverlay::createRasterOverlayOptions() const {
    CesiumRasterOverlays::RasterOverlayOptions options;
    options.ktx2TranscodeTargets = GltfUtil::getKtx2TranscodeTargets();
    options.ellipsoid = getEllipsoid();
    setRasterOverlayOptionsFromUsd(options);

    return options;
}

void OmniRasterOverlay::updateRasterOverlayOptions() const {
    const auto pRasterOverlay = getRasterOverlay();
    if (pRasterOverlay) {
        setRasterOverlayOptionsFromUsd(pRasterOverlay->getOptions());
    }
}

const CesiumGeospatial::Ellipsoid& OmniRasterOverlay::getEllipsoid() const {
    const auto& tilesets = _pContext->getAssetRegistry().getTilesets();
    for (const auto& pTileset : tilesets) {
        if (CppUtil::contains(pTileset->getRasterOverlayPaths(), _path)) {
            // Just use the first tileset's ellipsoid
            return pTileset->getEllipsoid();
        }
    }

    return CesiumGeospatial::Ellipsoid::WGS84;
}

void OmniRasterOverlay::setRasterOverlayOptionsFromUsd(CesiumRasterOverlays::RasterOverlayOptions& options) const {
    options.showCreditsOnScreen = getShowCreditsOnScreen();
    options.maximumScreenSpaceError = getMaximumScreenSpaceError();
    options.maximumTextureSize = getMaximumTextureSize();
    options.maximumSimultaneousTileLoads = getMaximumSimultaneousTileLoads();
    options.subTileCacheBytes = getSubTileCacheBytes();
}

} // namespace cesium::omniverse
