#include "cesium/omniverse/OmniTileset.h"

#include "cesium/omniverse/Broadcast.h"
#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/FabricMesh.h"
#include "cesium/omniverse/FabricPrepareRenderResources.h"
#include "cesium/omniverse/FabricUtil.h"
#include "cesium/omniverse/HttpAssetAccessor.h"
#include "cesium/omniverse/LoggerSink.h"
#include "cesium/omniverse/OmniImagery.h"
#include "cesium/omniverse/TaskProcessor.h"
#include "cesium/omniverse/UsdUtil.h"
#include "cesium/omniverse/Viewport.h"

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <Cesium3DTilesSelection/IonRasterOverlay.h>
#include <Cesium3DTilesSelection/Tileset.h>
#include <Cesium3DTilesSelection/ViewState.h>
#include <Cesium3DTilesSelection/ViewUpdateResult.h>
#include <CesiumUsdSchemas/imagery.h>
#include <CesiumUsdSchemas/tilesetAPI.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/stage.h>

namespace cesium::omniverse {

OmniTileset::OmniTileset(const pxr::SdfPath& tilesetPath)
    : _tilesetPath(tilesetPath)
    , _tilesetId(Context::instance().getNextTilesetId()) {
    reload();
}

OmniTileset::~OmniTileset() {}

pxr::SdfPath OmniTileset::getPath() const {
    return _tilesetPath;
}

std::string OmniTileset::getName() const {
    auto tileset = UsdUtil::getCesiumTileset(_tilesetPath);
    return tileset.GetPrim().GetName().GetString();
}

TilesetSourceType OmniTileset::getSourceType() const {
    auto tileset = UsdUtil::getCesiumTileset(_tilesetPath);

    pxr::TfToken sourceType;
    tileset.GetSourceTypeAttr().Get<pxr::TfToken>(&sourceType);

    if (sourceType == pxr::CesiumTokens->url) {
        return TilesetSourceType::URL;
    }

    return TilesetSourceType::ION;
}

std::string OmniTileset::getUrl() const {
    auto tileset = UsdUtil::getCesiumTileset(_tilesetPath);

    std::string url;
    tileset.GetUrlAttr().Get<std::string>(&url);

    return url;
}

int64_t OmniTileset::getIonAssetId() const {
    auto tileset = UsdUtil::getCesiumTileset(_tilesetPath);

    int64_t ionAssetId;
    tileset.GetIonAssetIdAttr().Get<int64_t>(&ionAssetId);

    return ionAssetId;
}

std::optional<CesiumIonClient::Token> OmniTileset::getIonAccessToken() const {
    auto tileset = UsdUtil::getCesiumTileset(_tilesetPath);

    std::string ionAccessToken;
    tileset.GetIonAccessTokenAttr().Get<std::string>(&ionAccessToken);

    if (ionAccessToken.empty()) {
        return Context::instance().getDefaultToken();
    }

    CesiumIonClient::Token t;
    t.token = ionAccessToken;

    return t;
}

float OmniTileset::getMaximumScreenSpaceError() const {
    auto tileset = UsdUtil::getCesiumTileset(_tilesetPath);

    float maximumScreenSpaceError;
    tileset.GetMaximumScreenSpaceErrorAttr().Get<float>(&maximumScreenSpaceError);

    return maximumScreenSpaceError;
}

bool OmniTileset::getPreloadAncestors() const {
    auto tileset = UsdUtil::getCesiumTileset(_tilesetPath);

    bool preloadAncestors;
    tileset.GetPreloadAncestorsAttr().Get<bool>(&preloadAncestors);

    return preloadAncestors;
}

bool OmniTileset::getPreloadSiblings() const {
    auto tileset = UsdUtil::getCesiumTileset(_tilesetPath);

    bool preloadSiblings;
    tileset.GetPreloadSiblingsAttr().Get<bool>(&preloadSiblings);

    return preloadSiblings;
}

bool OmniTileset::getForbidHoles() const {
    auto tileset = UsdUtil::getCesiumTileset(_tilesetPath);

    bool forbidHoles;
    tileset.GetForbidHolesAttr().Get<bool>(&forbidHoles);

    return forbidHoles;
}

uint32_t OmniTileset::getMaximumSimultaneousTileLoads() const {
    auto tileset = UsdUtil::getCesiumTileset(_tilesetPath);

    uint32_t maximumSimultaneousTileLoads;
    tileset.GetMaximumSimultaneousTileLoadsAttr().Get<uint32_t>(&maximumSimultaneousTileLoads);

    return maximumSimultaneousTileLoads;
}

uint64_t OmniTileset::getMaximumCachedBytes() const {
    auto tileset = UsdUtil::getCesiumTileset(_tilesetPath);

    uint64_t maximumCachedBytes;
    tileset.GetMaximumCachedBytesAttr().Get<uint64_t>(&maximumCachedBytes);

    return maximumCachedBytes;
}

uint32_t OmniTileset::getLoadingDescendantLimit() const {
    auto tileset = UsdUtil::getCesiumTileset(_tilesetPath);

    uint32_t loadingDescendantLimit;
    tileset.GetLoadingDescendantLimitAttr().Get<uint32_t>(&loadingDescendantLimit);

    return loadingDescendantLimit;
}

bool OmniTileset::getEnableFrustumCulling() const {
    auto tileset = UsdUtil::getCesiumTileset(_tilesetPath);

    bool enableFrustumCulling;
    tileset.GetEnableFrustumCullingAttr().Get<bool>(&enableFrustumCulling);

    return enableFrustumCulling;
}

bool OmniTileset::getEnableFogCulling() const {
    auto tileset = UsdUtil::getCesiumTileset(_tilesetPath);

    bool enableFogCulling;
    tileset.GetEnableFogCullingAttr().Get<bool>(&enableFogCulling);

    return enableFogCulling;
}

bool OmniTileset::getEnforceCulledScreenSpaceError() const {
    auto tileset = UsdUtil::getCesiumTileset(_tilesetPath);

    bool enforceCulledScreenSpaceError;
    tileset.GetEnforceCulledScreenSpaceErrorAttr().Get<bool>(&enforceCulledScreenSpaceError);

    return enforceCulledScreenSpaceError;
}

float OmniTileset::getCulledScreenSpaceError() const {
    auto tileset = UsdUtil::getCesiumTileset(_tilesetPath);

    float culledScreenSpaceError;
    tileset.GetCulledScreenSpaceErrorAttr().Get<float>(&culledScreenSpaceError);

    return culledScreenSpaceError;
}

bool OmniTileset::getSuspendUpdate() const {
    auto tileset = UsdUtil::getCesiumTileset(_tilesetPath);

    bool suspendUpdate;
    tileset.GetSuspendUpdateAttr().Get<bool>(&suspendUpdate);

    return suspendUpdate;
}

bool OmniTileset::getSmoothNormals() const {
    auto tileset = UsdUtil::getCesiumTileset(_tilesetPath);

    bool smoothNormals;
    tileset.GetSmoothNormalsAttr().Get<bool>(&smoothNormals);

    return smoothNormals;
}

bool OmniTileset::getShowCreditsOnScreen() const {
    auto tileset = UsdUtil::getCesiumTileset(_tilesetPath);

    bool showCreditsOnScreen;
    tileset.GetShowCreditsOnScreenAttr().Get<bool>(&showCreditsOnScreen);

    return showCreditsOnScreen;
}

int64_t OmniTileset::getTilesetId() const {
    return _tilesetId;
}

void OmniTileset::reload() {
    _renderResourcesPreparer = std::make_shared<FabricPrepareRenderResources>(*this);
    auto& context = Context::instance();
    const auto asyncSystem = CesiumAsync::AsyncSystem(context.getTaskProcessor());
    const auto externals = Cesium3DTilesSelection::TilesetExternals{
        context.getHttpAssetAccessor(),
        _renderResourcesPreparer,
        std::move(asyncSystem),
        context.getCreditSystem(),
        context.getLogger()};

    const auto url = getUrl();
    const auto tilesetPath = getPath();
    const auto ionAssetId = getIonAssetId();
    const auto ionAccessToken = getIonAccessToken();
    const auto name = getName();

    Cesium3DTilesSelection::TilesetOptions options;

    options.maximumScreenSpaceError = static_cast<double>(getMaximumScreenSpaceError());
    options.preloadAncestors = getPreloadAncestors();
    options.preloadSiblings = getPreloadSiblings();
    options.forbidHoles = getForbidHoles();
    options.maximumSimultaneousTileLoads = getMaximumSimultaneousTileLoads();
    options.maximumCachedBytes = static_cast<int64_t>(getMaximumCachedBytes());
    options.loadingDescendantLimit = getLoadingDescendantLimit();
    options.enableFrustumCulling = getEnableFrustumCulling();
    options.enableFogCulling = getEnableFogCulling();
    options.enforceCulledScreenSpaceError = getEnforceCulledScreenSpaceError();
    options.culledScreenSpaceError = getCulledScreenSpaceError();
    options.showCreditsOnScreen = getShowCreditsOnScreen();

    options.loadErrorCallback =
        [tilesetPath, ionAssetId, name](const Cesium3DTilesSelection::TilesetLoadFailureDetails& error) {
            // Check for a 401 connecting to Cesium ion, which means the token is invalid
            // (or perhaps the asset ID is). Also check for a 404, because ion returns 404
            // when the token is valid but not authorized for the asset.
            if (error.type == Cesium3DTilesSelection::TilesetLoadType::CesiumIon &&
                (error.statusCode == 401 || error.statusCode == 404)) {
                Broadcast::showTroubleshooter(tilesetPath, ionAssetId, name, 0, "", error.message);
            }

            CESIUM_LOG_ERROR(error.message);
        };

    _pViewUpdateResult = nullptr;

    if (getSourceType() == TilesetSourceType::URL) {
        _tileset = std::make_unique<Cesium3DTilesSelection::Tileset>(externals, url, options);
    } else if (!ionAccessToken.has_value()) {
        // This happens when adding a blank tileset.
        _tileset = std::make_unique<Cesium3DTilesSelection::Tileset>(externals, 0, "", options);
    } else {
        _tileset = std::make_unique<Cesium3DTilesSelection::Tileset>(
            externals, ionAssetId, ionAccessToken.value().token, options);
    }

    // Add imagery
    for (const auto& imagery : UsdUtil::getChildCesiumImageryPrims(_tilesetPath)) {
        addImageryIon(imagery.GetPath());
    }
}

void OmniTileset::addImageryIon(const pxr::SdfPath& imageryPath) {
    const OmniImagery imagery(imageryPath);
    const auto imageryIonAssetId = imagery.getIonAssetId();
    const auto imageryIonAccessToken = imagery.getIonAccessToken();

    if (!imageryIonAccessToken.has_value()) {
        // If we don't have an access token available there's no point in adding the imagery.
        return;
    }

    const auto imageryName = imagery.getName();

    const auto tilesetPath = getPath();
    const auto tilesetIonAssetId = getIonAssetId();
    const auto tilesetName = getName();

    Cesium3DTilesSelection::RasterOverlayOptions options;
    options.loadErrorCallback = [tilesetPath, tilesetIonAssetId, tilesetName, imageryIonAssetId, imageryName](
                                    const Cesium3DTilesSelection::RasterOverlayLoadFailureDetails& error) {
        // Check for a 401 connecting to Cesium ion, which means the token is invalid
        // (or perhaps the asset ID is). Also check for a 404, because ion returns 404
        // when the token is valid but not authorized for the asset.
        auto statusCode = error.pRequest && error.pRequest->response() ? error.pRequest->response()->statusCode() : 0;

        if (error.type == Cesium3DTilesSelection::RasterOverlayLoadType::CesiumIon &&
            (statusCode == 401 || statusCode == 404)) {
            Broadcast::showTroubleshooter(
                tilesetPath, tilesetIonAssetId, tilesetName, imageryIonAssetId, imageryName, error.message);
        }

        CESIUM_LOG_ERROR(error.message);
    };

    // The name passed to IonRasterOverlay needs to uniquely identify this imagery otherwise texture caching may break
    const auto uniqueName = fmt::format("imagery_ion_{}", imageryIonAssetId);
    const auto ionRasterOverlay = new Cesium3DTilesSelection::IonRasterOverlay(
        uniqueName, imageryIonAssetId, imageryIonAccessToken.value().token, options);
    _tileset->getOverlays().add(ionRasterOverlay);
}

void OmniTileset::onUpdateFrame(const std::vector<Viewport>& viewports) {
    if (!UsdUtil::primExists(_tilesetPath)) {
        // TfNotice can be slow, and sometimes we get a frame or two before we actually get a chance to react on it.
        //   This guard prevents us from crashing if the prim no longer exists.
        return;
    }

    updateTransform();
    updateView(viewports);
}

void OmniTileset::updateTransform() {
    // computeEcefToUsdTransformForPrim is slightly expensive operations to do every frame but it is simple
    // and exhaustive. E.g. it reacts to USD scene graph changes, up-axis changes, meters-per-unit changes, and georeference origin changes
    // without us needing to subscribe to any events.
    //
    // The faster approach would be to load the tileset USD prim into Fabric (via usdrt::UsdStage::GetPrimAtPath)
    // and subscribe to change events for _worldPosition, _worldOrientation, _worldScale.
    // Alternatively, we could register a listener with Tf::Notice but this has the downside of only notifying us
    // about changes to the current prim and not its ancestor prims. Also Tf::Notice may notify us in a thread other
    // than the main thread and we would have to be careful to synchronize updates to Fabric in the main thread.

    const auto georeferenceOrigin = Context::instance().getGeoreferenceOrigin();
    const auto ecefToUsdTransform = UsdUtil::computeEcefToUsdTransformForPrim(georeferenceOrigin, _tilesetPath);

    // Check for transform changes and update prims accordingly
    if (ecefToUsdTransform != _ecefToUsdTransform) {
        _ecefToUsdTransform = ecefToUsdTransform;
        FabricUtil::setTilesetTransform(_tilesetId, ecefToUsdTransform);
    }
}

void OmniTileset::updateView(const std::vector<Viewport>& viewports) {
    if (!getSuspendUpdate()) {
        // Go ahead and select some tiles
        const auto& georeferenceOrigin = Context::instance().getGeoreferenceOrigin();

        _viewStates.clear();
        for (const auto& viewport : viewports) {
            _viewStates.emplace_back(UsdUtil::computeViewState(georeferenceOrigin, _tilesetPath, viewport));
        }

        _pViewUpdateResult = &_tileset->updateView(_viewStates);
    }

    if (!_pViewUpdateResult) {
        // No tiles have ever been selected. Return early.
        return;
    }

    const auto visible = UsdUtil::isPrimVisible(_tilesetPath);

    // Hide tiles that we no longer need
    for (const auto tile : _pViewUpdateResult->tilesFadingOut) {
        if (tile->getState() == Cesium3DTilesSelection::TileLoadState::Done) {
            const auto pRenderContent = tile->getContent().getRenderContent();
            if (pRenderContent) {
                const auto pRenderResources = pRenderContent->getRenderResources();
                if (pRenderResources) {
                    const auto pTileRenderResources = reinterpret_cast<TileRenderResources*>(pRenderResources);
                    for (const auto& fabricMesh : pTileRenderResources->fabricMeshes) {
                        fabricMesh->setVisibility(false);
                    }
                }
            }
        }
    }

    // Update visibility for selected tiles
    for (const auto tile : _pViewUpdateResult->tilesToRenderThisFrame) {
        if (tile->getState() == Cesium3DTilesSelection::TileLoadState::Done) {
            const auto pRenderContent = tile->getContent().getRenderContent();
            if (pRenderContent) {
                const auto pRenderResources = pRenderContent->getRenderResources();
                if (pRenderResources) {
                    const auto pTileRenderResources = reinterpret_cast<TileRenderResources*>(pRenderResources);
                    for (const auto& fabricMesh : pTileRenderResources->fabricMeshes) {
                        fabricMesh->setVisibility(visible);
                    }
                }
            }
        }
    }
}

} // namespace cesium::omniverse
