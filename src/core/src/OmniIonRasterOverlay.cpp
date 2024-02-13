#include "cesium/omniverse/OmniIonRasterOverlay.h"

#include "cesium/omniverse/AssetRegistry.h"
#include "cesium/omniverse/Broadcast.h"
#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/Logger.h"
#include "cesium/omniverse/OmniIonServer.h"
#include "cesium/omniverse/UsdUtil.h"

#include <CesiumAsync/IAssetResponse.h>
#include <CesiumIonClient/Token.h>
#include <CesiumRasterOverlays/IonRasterOverlay.h>
#include <CesiumUsdSchemas/ionRasterOverlay.h>
#include <CesiumUtility/IntrusivePointer.h>

namespace cesium::omniverse {

namespace {} // namespace

OmniIonRasterOverlay::OmniIonRasterOverlay(Context* pContext, const pxr::SdfPath& path)
    : OmniRasterOverlay(pContext, path) {
    reload();
}

int64_t OmniIonRasterOverlay::getIonAssetId() const {
    const auto cesiumIonRasterOverlay = UsdUtil::getCesiumIonRasterOverlay(_pContext->getUsdStage(), _path);

    int64_t ionAssetId;
    cesiumIonRasterOverlay.GetIonAssetIdAttr().Get(&ionAssetId);

    return ionAssetId;
}

CesiumIonClient::Token OmniIonRasterOverlay::getIonAccessToken() const {
    const auto cesiumIonRasterOverlay = UsdUtil::getCesiumIonRasterOverlay(_pContext->getUsdStage(), _path);

    std::string ionAccessToken;
    cesiumIonRasterOverlay.GetIonAccessTokenAttr().Get(&ionAccessToken);

    if (!ionAccessToken.empty()) {
        CesiumIonClient::Token t;
        t.token = ionAccessToken;
        return t;
    }

    const auto ionServerPath = getResolvedIonServerPath();

    if (ionServerPath.IsEmpty()) {
        return {};
    }

    const auto pIonServer = _pContext->getAssetRegistry().getIonServer(ionServerPath);

    if (!pIonServer) {
        return {};
    }

    return pIonServer->getToken();
}

std::string OmniIonRasterOverlay::getIonApiUrl() const {
    const auto ionServerPath = getResolvedIonServerPath();

    if (ionServerPath.IsEmpty()) {
        return {};
    }

    const auto pIonServer = _pContext->getAssetRegistry().getIonServer(ionServerPath);

    if (!pIonServer) {
        return {};
    }

    return pIonServer->getIonServerApiUrl();
}

pxr::SdfPath OmniIonRasterOverlay::getResolvedIonServerPath() const {
    const auto cesiumIonRasterOverlay = UsdUtil::getCesiumIonRasterOverlay(_pContext->getUsdStage(), _path);

    pxr::SdfPathVector targets;
    cesiumIonRasterOverlay.GetIonServerBindingRel().GetForwardedTargets(&targets);

    if (!targets.empty()) {
        return targets.front();
    }

    // Fall back to using the first ion server if there's no explicit binding
    const auto pIonServer = _pContext->getAssetRegistry().getFirstIonServer();
    if (pIonServer) {
        return pIonServer->getPath();
    }

    return {};
}

CesiumRasterOverlays::RasterOverlay* OmniIonRasterOverlay::getRasterOverlay() const {
    return _pIonRasterOverlay.get();
}

void OmniIonRasterOverlay::reload() {
    const auto rasterOverlayIonAssetId = getIonAssetId();
    const auto rasterOverlayIonAccessToken = getIonAccessToken();
    const auto rasterOverlayIonApiUrl = getIonApiUrl();

    if (rasterOverlayIonAssetId <= 0 || rasterOverlayIonAccessToken.token.empty() || rasterOverlayIonApiUrl.empty()) {
        return;
    }

    const auto rasterOverlayName = UsdUtil::getName(_pContext->getUsdStage(), _path);

    auto options = getUsdOptions();

    options.loadErrorCallback = [this, rasterOverlayIonAssetId, rasterOverlayName](
                                    const CesiumRasterOverlays::RasterOverlayLoadFailureDetails& error) {
        // Check for a 401 connecting to Cesium ion, which means the token is invalid
        // (or perhaps the asset ID is). Also check for a 404, because ion returns 404
        // when the token is valid but not authorized for the asset.
        const auto statusCode =
            error.pRequest && error.pRequest->response() ? error.pRequest->response()->statusCode() : 0;

        if (error.type == CesiumRasterOverlays::RasterOverlayLoadType::CesiumIon &&
            (statusCode == 401 || statusCode == 404)) {
            // TODO: this probably doesn't work without tileset info
            Broadcast::showTroubleshooter({}, 0, "", rasterOverlayIonAssetId, rasterOverlayName, error.message);
        }

        _pContext->getLogger()->error(error.message);
    };

    _pIonRasterOverlay = new CesiumRasterOverlays::IonRasterOverlay(
        rasterOverlayName, rasterOverlayIonAssetId, rasterOverlayIonAccessToken.token, options, rasterOverlayIonApiUrl);
}

} // namespace cesium::omniverse
