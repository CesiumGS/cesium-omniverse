#include "cesium/omniverse/OmniIonImagery.h"

#include "cesium/omniverse/AssetRegistry.h"
#include "cesium/omniverse/Broadcast.h"
#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/GltfUtil.h"
#include "cesium/omniverse/Logger.h"
#include "cesium/omniverse/OmniIonServer.h"
#include "cesium/omniverse/UsdUtil.h"

#include <CesiumAsync/IAssetResponse.h>
#include <CesiumIonClient/Token.h>
#include <CesiumRasterOverlays/IonRasterOverlay.h>
#include <CesiumUsdSchemas/ionImagery.h>
#include <CesiumUtility/IntrusivePointer.h>

namespace cesium::omniverse {

namespace {} // namespace

OmniIonImagery::OmniIonImagery(Context* pContext, const pxr::SdfPath& path)
    : OmniImagery(pContext, path) {
    reload();
}

int64_t OmniIonImagery::getIonAssetId() const {
    const auto cesiumIonImagery = UsdUtil::getCesiumIonImagery(_pContext->getUsdStage(), _path);

    int64_t ionAssetId;
    cesiumIonImagery.GetIonAssetIdAttr().Get(&ionAssetId);

    return ionAssetId;
}

CesiumIonClient::Token OmniIonImagery::getIonAccessToken() const {
    const auto cesiumIonImagery = UsdUtil::getCesiumIonImagery(_pContext->getUsdStage(), _path);

    std::string ionAccessToken;
    cesiumIonImagery.GetIonAccessTokenAttr().Get(&ionAccessToken);

    if (!ionAccessToken.empty()) {
        CesiumIonClient::Token t;
        t.token = ionAccessToken;
        return t;
    }

    const auto ionServerPath = getIonServerPath();

    if (ionServerPath.IsEmpty()) {
        return {};
    }

    const auto pIonServer = _pContext->getAssetRegistry().getIonServer(ionServerPath);

    if (!pIonServer) {
        return {};
    }

    return pIonServer->getToken();
}

std::string OmniIonImagery::getIonApiUrl() const {
    const auto ionServerPath = getIonServerPath();

    if (ionServerPath.IsEmpty()) {
        return {};
    }

    const auto pIonServer = _pContext->getAssetRegistry().getIonServer(ionServerPath);

    if (!pIonServer) {
        return {};
    }

    return pIonServer->getIonServerApiUrl();
}

pxr::SdfPath OmniIonImagery::getIonServerPath() const {
    const auto cesiumIonImagery = UsdUtil::getCesiumIonImagery(_pContext->getUsdStage(), _path);

    pxr::SdfPathVector targets;
    cesiumIonImagery.GetIonServerBindingRel().GetForwardedTargets(&targets);

    if (targets.empty()) {
        return {};
    }

    return targets.front();
}

CesiumRasterOverlays::RasterOverlay* OmniIonImagery::getRasterOverlay() const {
    return _pIonRasterOverlay.get();
}

void OmniIonImagery::reload() {
    const auto imageryIonAssetId = getIonAssetId();
    const auto imageryIonAccessToken = getIonAccessToken();
    const auto imageryIonApiUrl = getIonApiUrl();

    if (imageryIonAssetId <= 0 || imageryIonAccessToken.token.empty() || imageryIonApiUrl.empty()) {
        return;
    }

    const auto imageryName = UsdUtil::getName(_pContext->getUsdStage(), _path);

    CesiumRasterOverlays::RasterOverlayOptions options;
    options.showCreditsOnScreen = getShowCreditsOnScreen();
    options.ktx2TranscodeTargets = GltfUtil::getKtx2TranscodeTargets();

    options.loadErrorCallback =
        [this, imageryIonAssetId, imageryName](const CesiumRasterOverlays::RasterOverlayLoadFailureDetails& error) {
            // Check for a 401 connecting to Cesium ion, which means the token is invalid
            // (or perhaps the asset ID is). Also check for a 404, because ion returns 404
            // when the token is valid but not authorized for the asset.
            const auto statusCode =
                error.pRequest && error.pRequest->response() ? error.pRequest->response()->statusCode() : 0;

            if (error.type == CesiumRasterOverlays::RasterOverlayLoadType::CesiumIon &&
                (statusCode == 401 || statusCode == 404)) {
                // TODO: this probably doesn't work without tileset info
                Broadcast::showTroubleshooter({}, 0, "", imageryIonAssetId, imageryName, error.message);
            }

            _pContext->getLogger()->error(error.message);
        };

    _pIonRasterOverlay = new CesiumRasterOverlays::IonRasterOverlay(
        imageryName, imageryIonAssetId, imageryIonAccessToken.token, options, imageryIonApiUrl);
}

void OmniIonImagery::setIonServerPath(const pxr::SdfPath& ionServerPath) {
    if (ionServerPath.IsEmpty()) {
        return;
    }

    const auto cesiumIonImagery = UsdUtil::getCesiumIonImagery(_pContext->getUsdStage(), _path);
    cesiumIonImagery.GetIonServerBindingRel().SetTargets({ionServerPath});
}

} // namespace cesium::omniverse
