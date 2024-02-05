#include "cesium/omniverse/OmniTileset.h"

#include "cesium/omniverse/AssetRegistry.h"
#include "cesium/omniverse/Broadcast.h"
#include "cesium/omniverse/CesiumIonSession.h"
#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/FabricGeometry.h"
#include "cesium/omniverse/FabricMaterial.h"
#include "cesium/omniverse/FabricMesh.h"
#include "cesium/omniverse/FabricPrepareRenderResources.h"
#include "cesium/omniverse/FabricRenderResources.h"
#include "cesium/omniverse/FabricUtil.h"
#include "cesium/omniverse/GltfUtil.h"
#include "cesium/omniverse/HttpAssetAccessor.h"
#include "cesium/omniverse/Logger.h"
#include "cesium/omniverse/OmniCartographicPolygon.h"
#include "cesium/omniverse/OmniGeoreference.h"
#include "cesium/omniverse/OmniGlobeAnchor.h"
#include "cesium/omniverse/OmniIonRasterOverlay.h"
#include "cesium/omniverse/OmniIonServer.h"
#include "cesium/omniverse/OmniPolygonRasterOverlay.h"
#include "cesium/omniverse/TaskProcessor.h"
#include "cesium/omniverse/TilesetStatistics.h"
#include "cesium/omniverse/UsdUtil.h"
#include "cesium/omniverse/Viewport.h"

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <Cesium3DTilesSelection/Tileset.h>
#include <Cesium3DTilesSelection/ViewState.h>
#include <Cesium3DTilesSelection/ViewUpdateResult.h>
#include <CesiumGeospatial/CartographicPolygon.h>
#include <CesiumRasterOverlays/IonRasterOverlay.h>
#include <CesiumRasterOverlays/RasterizedPolygonsOverlay.h>
#include <CesiumUsdSchemas/rasterOverlay.h>
#include <CesiumUsdSchemas/tileset.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/boundable.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>

namespace cesium::omniverse {

namespace {

void forEachFabricMaterial(
    Cesium3DTilesSelection::Tileset* pTileset,
    const std::function<void(FabricMaterial& fabricMaterial)>& callback) {
    pTileset->forEachLoadedTile([&callback](Cesium3DTilesSelection::Tile& tile) {
        if (tile.getState() != Cesium3DTilesSelection::TileLoadState::Done) {
            return;
        }
        const auto& content = tile.getContent();
        const auto pRenderContent = content.getRenderContent();
        if (!pRenderContent) {
            return;
        }
        const auto pFabricRenderResources = static_cast<FabricRenderResources*>(pRenderContent->getRenderResources());
        if (!pFabricRenderResources) {
            return;
        }
        for (const auto& fabricMesh : pFabricRenderResources->fabricMeshes) {
            if (fabricMesh.pMaterial) {
                callback(*fabricMesh.pMaterial.get());
            }
        }
    });
}

} // namespace

OmniTileset::OmniTileset(Context* pContext, const pxr::SdfPath& path, int64_t tilesetId)
    : _pContext(pContext)
    , _path(path)
    , _tilesetId(tilesetId) {
    reload();
}

OmniTileset::~OmniTileset() {
    destroyNativeTileset();
}

const pxr::SdfPath& OmniTileset::getPath() const {
    return _path;
}

int64_t OmniTileset::getTilesetId() const {
    return _tilesetId;
}

TilesetStatistics OmniTileset::getStatistics() const {
    TilesetStatistics statistics;

    statistics.tilesetCachedBytes = static_cast<uint64_t>(_pTileset->getTotalDataBytes());
    statistics.tilesLoaded = static_cast<uint64_t>(_pTileset->getNumberOfTilesLoaded());

    if (_pViewUpdateResult) {
        statistics.tilesVisited = static_cast<uint64_t>(_pViewUpdateResult->tilesVisited);
        statistics.culledTilesVisited = static_cast<uint64_t>(_pViewUpdateResult->culledTilesVisited);
        statistics.tilesRendered = static_cast<uint64_t>(_pViewUpdateResult->tilesToRenderThisFrame.size());
        statistics.tilesCulled = static_cast<uint64_t>(_pViewUpdateResult->tilesCulled);
        statistics.maxDepthVisited = static_cast<uint64_t>(_pViewUpdateResult->maxDepthVisited);
        statistics.tilesLoadingWorker = static_cast<uint64_t>(_pViewUpdateResult->workerThreadTileLoadQueueLength);
        statistics.tilesLoadingMain = static_cast<uint64_t>(_pViewUpdateResult->mainThreadTileLoadQueueLength);
    }

    return statistics;
}

TilesetSourceType OmniTileset::getSourceType() const {
    const auto cesiumTileset = UsdUtil::getCesiumTileset(_pContext->getUsdStage(), _path);

    pxr::TfToken sourceType;
    cesiumTileset.GetSourceTypeAttr().Get(&sourceType);

    if (sourceType == pxr::CesiumTokens->url) {
        return TilesetSourceType::URL;
    }

    return TilesetSourceType::ION;
}

std::string OmniTileset::getUrl() const {
    const auto cesiumTileset = UsdUtil::getCesiumTileset(_pContext->getUsdStage(), _path);

    std::string url;
    cesiumTileset.GetUrlAttr().Get(&url);

    return url;
}

int64_t OmniTileset::getIonAssetId() const {
    const auto cesiumTileset = UsdUtil::getCesiumTileset(_pContext->getUsdStage(), _path);

    int64_t ionAssetId;
    cesiumTileset.GetIonAssetIdAttr().Get(&ionAssetId);

    return ionAssetId;
}

CesiumIonClient::Token OmniTileset::getIonAccessToken() const {
    const auto cesiumTileset = UsdUtil::getCesiumTileset(_pContext->getUsdStage(), _path);

    std::string ionAccessToken;
    cesiumTileset.GetIonAccessTokenAttr().Get(&ionAccessToken);

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

std::string OmniTileset::getIonApiUrl() const {
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

pxr::SdfPath OmniTileset::getResolvedIonServerPath() const {
    const auto cesiumTileset = UsdUtil::getCesiumTileset(_pContext->getUsdStage(), _path);

    pxr::SdfPathVector targets;
    cesiumTileset.GetIonServerBindingRel().GetForwardedTargets(&targets);

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

double OmniTileset::getMaximumScreenSpaceError() const {
    const auto cesiumTileset = UsdUtil::getCesiumTileset(_pContext->getUsdStage(), _path);

    float maximumScreenSpaceError;
    cesiumTileset.GetMaximumScreenSpaceErrorAttr().Get(&maximumScreenSpaceError);

    return static_cast<double>(maximumScreenSpaceError);
}

bool OmniTileset::getPreloadAncestors() const {
    const auto cesiumTileset = UsdUtil::getCesiumTileset(_pContext->getUsdStage(), _path);

    bool preloadAncestors;
    cesiumTileset.GetPreloadAncestorsAttr().Get(&preloadAncestors);

    return preloadAncestors;
}

bool OmniTileset::getPreloadSiblings() const {
    const auto cesiumTileset = UsdUtil::getCesiumTileset(_pContext->getUsdStage(), _path);

    bool preloadSiblings;
    cesiumTileset.GetPreloadSiblingsAttr().Get(&preloadSiblings);

    return preloadSiblings;
}

bool OmniTileset::getForbidHoles() const {
    const auto cesiumTileset = UsdUtil::getCesiumTileset(_pContext->getUsdStage(), _path);

    bool forbidHoles;
    cesiumTileset.GetForbidHolesAttr().Get(&forbidHoles);

    return forbidHoles;
}

uint32_t OmniTileset::getMaximumSimultaneousTileLoads() const {
    const auto cesiumTileset = UsdUtil::getCesiumTileset(_pContext->getUsdStage(), _path);

    uint32_t maximumSimultaneousTileLoads;
    cesiumTileset.GetMaximumSimultaneousTileLoadsAttr().Get(&maximumSimultaneousTileLoads);

    return maximumSimultaneousTileLoads;
}

uint64_t OmniTileset::getMaximumCachedBytes() const {
    const auto cesiumTileset = UsdUtil::getCesiumTileset(_pContext->getUsdStage(), _path);

    uint64_t maximumCachedBytes;
    cesiumTileset.GetMaximumCachedBytesAttr().Get(&maximumCachedBytes);

    return maximumCachedBytes;
}

uint32_t OmniTileset::getLoadingDescendantLimit() const {
    const auto cesiumTileset = UsdUtil::getCesiumTileset(_pContext->getUsdStage(), _path);

    uint32_t loadingDescendantLimit;
    cesiumTileset.GetLoadingDescendantLimitAttr().Get(&loadingDescendantLimit);

    return loadingDescendantLimit;
}

bool OmniTileset::getEnableFrustumCulling() const {
    const auto cesiumTileset = UsdUtil::getCesiumTileset(_pContext->getUsdStage(), _path);

    bool enableFrustumCulling;
    cesiumTileset.GetEnableFrustumCullingAttr().Get(&enableFrustumCulling);

    return enableFrustumCulling;
}

bool OmniTileset::getEnableFogCulling() const {
    const auto cesiumTileset = UsdUtil::getCesiumTileset(_pContext->getUsdStage(), _path);

    bool enableFogCulling;
    cesiumTileset.GetEnableFogCullingAttr().Get(&enableFogCulling);

    return enableFogCulling;
}

bool OmniTileset::getEnforceCulledScreenSpaceError() const {
    const auto cesiumTileset = UsdUtil::getCesiumTileset(_pContext->getUsdStage(), _path);

    bool enforceCulledScreenSpaceError;
    cesiumTileset.GetEnforceCulledScreenSpaceErrorAttr().Get(&enforceCulledScreenSpaceError);

    return enforceCulledScreenSpaceError;
}

double OmniTileset::getMainThreadLoadingTimeLimit() const {
    const auto cesiumTileset = UsdUtil::getCesiumTileset(_pContext->getUsdStage(), _path);

    float mainThreadLoadingTimeLimit;
    cesiumTileset.GetMainThreadLoadingTimeLimitAttr().Get(&mainThreadLoadingTimeLimit);

    return static_cast<double>(mainThreadLoadingTimeLimit);
}

double OmniTileset::getCulledScreenSpaceError() const {
    const auto cesiumTileset = UsdUtil::getCesiumTileset(_pContext->getUsdStage(), _path);

    float culledScreenSpaceError;
    cesiumTileset.GetCulledScreenSpaceErrorAttr().Get(&culledScreenSpaceError);

    return static_cast<double>(culledScreenSpaceError);
}

bool OmniTileset::getSuspendUpdate() const {
    const auto cesiumTileset = UsdUtil::getCesiumTileset(_pContext->getUsdStage(), _path);

    bool suspendUpdate;
    cesiumTileset.GetSuspendUpdateAttr().Get(&suspendUpdate);

    return suspendUpdate;
}

bool OmniTileset::getSmoothNormals() const {
    const auto cesiumTileset = UsdUtil::getCesiumTileset(_pContext->getUsdStage(), _path);

    bool smoothNormals;
    cesiumTileset.GetSmoothNormalsAttr().Get(&smoothNormals);

    return smoothNormals;
}

bool OmniTileset::getShowCreditsOnScreen() const {
    const auto cesiumTileset = UsdUtil::getCesiumTileset(_pContext->getUsdStage(), _path);

    bool showCreditsOnScreen;
    cesiumTileset.GetShowCreditsOnScreenAttr().Get(&showCreditsOnScreen);

    return showCreditsOnScreen;
}

pxr::SdfPath OmniTileset::getResolvedGeoreferencePath() const {
    const auto cesiumTileset = UsdUtil::getCesiumTileset(_pContext->getUsdStage(), _path);

    pxr::SdfPathVector targets;
    cesiumTileset.GetGeoreferenceBindingRel().GetForwardedTargets(&targets);

    if (!targets.empty()) {
        return targets.front();
    }

    // Fall back to using the first georeference if there's no explicit binding
    const auto pGeoreference = _pContext->getAssetRegistry().getFirstGeoreference();
    if (pGeoreference) {
        return pGeoreference->getPath();
    }

    return {};
}

pxr::SdfPath OmniTileset::getMaterialPath() const {
    const auto cesiumTileset = UsdUtil::getCesiumTileset(_pContext->getUsdStage(), _path);

    const auto materialBindingApi = pxr::UsdShadeMaterialBindingAPI(cesiumTileset);
    const auto materialBinding = materialBindingApi.GetDirectBinding();
    const auto& materialPath = materialBinding.GetMaterialPath();

    return materialPath;
}

glm::dvec3 OmniTileset::getDisplayColor() const {
    const auto cesiumTileset = UsdUtil::getCesiumTileset(_pContext->getUsdStage(), _path);

    pxr::VtVec3fArray displayColorArray;
    cesiumTileset.GetDisplayColorAttr().Get(&displayColorArray);

    if (displayColorArray.size() == 0) {
        return {1.0, 1.0, 1.0};
    }

    const auto& displayColor = displayColorArray[0];
    return {
        static_cast<double>(displayColor[0]),
        static_cast<double>(displayColor[1]),
        static_cast<double>(displayColor[2]),
    };
}

double OmniTileset::getDisplayOpacity() const {
    const auto cesiumTileset = UsdUtil::getCesiumTileset(_pContext->getUsdStage(), _path);

    pxr::VtFloatArray displayOpacityArray;
    cesiumTileset.GetDisplayOpacityAttr().Get(&displayOpacityArray);

    if (displayOpacityArray.size() == 0) {
        return 1.0;
    }

    return static_cast<double>(displayOpacityArray[0]);
}

std::vector<pxr::SdfPath> OmniTileset::getRasterOverlayPaths() const {
    const auto cesiumTileset = UsdUtil::getCesiumTileset(_pContext->getUsdStage(), _path);

    pxr::SdfPathVector targets;
    cesiumTileset.GetRasterOverlayBindingRel().GetForwardedTargets(&targets);

    return targets;
}

void OmniTileset::updateTilesetOptions() {
    auto& options = _pTileset->getOptions();
    options.maximumScreenSpaceError = getMaximumScreenSpaceError();
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
    options.mainThreadLoadingTimeLimit = getMainThreadLoadingTimeLimit();
    options.showCreditsOnScreen = getShowCreditsOnScreen();
}

void OmniTileset::reload() {
    destroyNativeTileset();

    _pRenderResourcesPreparer = std::make_shared<FabricPrepareRenderResources>(_pContext, this);
    const auto externals = Cesium3DTilesSelection::TilesetExternals{
        _pContext->getHttpAssetAccessor(),
        _pRenderResourcesPreparer,
        _pContext->getAsyncSystem(),
        _pContext->getCreditSystem(),
        _pContext->getLogger()};

    const auto sourceType = getSourceType();
    const auto url = getUrl();
    const auto tilesetPath = getPath();
    const auto ionAssetId = getIonAssetId();
    const auto ionAccessToken = getIonAccessToken();
    const auto ionApiUrl = getIonApiUrl();
    const auto name = UsdUtil::getName(_pContext->getUsdStage(), _path);

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
    options.mainThreadLoadingTimeLimit = getMainThreadLoadingTimeLimit();
    options.showCreditsOnScreen = getShowCreditsOnScreen();
    options.excluders = std::vector<std::shared_ptr<Cesium3DTilesSelection::ITileExcluder>>();

    options.loadErrorCallback =
        [this, tilesetPath, ionAssetId, name](const Cesium3DTilesSelection::TilesetLoadFailureDetails& error) {
            // Check for a 401 connecting to Cesium ion, which means the token is invalid
            // (or perhaps the asset ID is). Also check for a 404, because ion returns 404
            // when the token is valid but not authorized for the asset.
            if (error.type == Cesium3DTilesSelection::TilesetLoadType::CesiumIon &&
                (error.statusCode == 401 || error.statusCode == 404)) {
                Broadcast::showTroubleshooter(tilesetPath, ionAssetId, name, 0, "", error.message);
            }

            _pContext->getLogger()->error(error.message);
        };

    options.contentOptions.ktx2TranscodeTargets = GltfUtil::getKtx2TranscodeTargets();

    _pViewUpdateResult = nullptr;
    _extentSet = false;
    _activeLoading = false;

    switch (sourceType) {
        case TilesetSourceType::ION:
            if (ionAssetId <= 0 || ionAccessToken.token.empty() || ionApiUrl.empty()) {
                _pTileset = std::make_unique<Cesium3DTilesSelection::Tileset>(externals, 0, "", options);
            } else {
                _pTileset = std::make_unique<Cesium3DTilesSelection::Tileset>(
                    externals, ionAssetId, ionAccessToken.token, options, ionApiUrl);
            }
            break;
        case TilesetSourceType::URL:
            _pTileset = std::make_unique<Cesium3DTilesSelection::Tileset>(externals, url, options);
            break;
    }

    const auto cesiumTileset = UsdUtil::getCesiumTileset(_pContext->getUsdStage(), _path);
    const auto rasterOverlayPaths = getRasterOverlayPaths();

    for (const auto& rasterOverlayPath : rasterOverlayPaths) {
        const auto pRasterOverlay = _pContext->getAssetRegistry().getRasterOverlay(rasterOverlayPath);
        if (pRasterOverlay) {
            const auto pNativeRasterOverlay = pRasterOverlay->getRasterOverlay();
            if (pNativeRasterOverlay) {
                _pTileset->getOverlays().add(pNativeRasterOverlay);
            }
        }
    }

    const auto& allOmniPolygonRasterOverlays = _pContext->getAssetRegistry().getPolygonRasterOverlays();
    const auto& boundNativeRasterOverlays =
        _pTileset->getOverlays().getOverlays(); // get RasterOverlays from RasterOverlayCollection
    for (const auto& pOmniPolygonRasterOverlay : allOmniPolygonRasterOverlays) {
        const auto& pNativeOverlay = pOmniPolygonRasterOverlay->getRasterOverlay();

        auto matchIterator = std::find_if(
            boundNativeRasterOverlays.begin(),
            boundNativeRasterOverlays.end(),
            [&pNativeOverlay](const auto& possibleMatch) { return possibleMatch.get() == pNativeOverlay; });

        if (matchIterator == boundNativeRasterOverlays.end()) {
            continue;
        }

        const auto exclude = pOmniPolygonRasterOverlay->getExcludeSelectedTiles();
        if (exclude) {
            const auto excluder = pOmniPolygonRasterOverlay->getExcluder();
            if (excluder) {
                _pTileset->getOptions().excluders.push_back(excluder);
            }
        }
    }
}

pxr::SdfPath OmniTileset::getRasterOverlayPath(const CesiumRasterOverlays::RasterOverlay& rasterOverlay) const {
    const auto rasterOverlayPaths = getRasterOverlayPaths();

    for (const auto& rasterOverlayPath : rasterOverlayPaths) {
        const auto pRasterOverlay = _pContext->getAssetRegistry().getRasterOverlay(rasterOverlayPath);
        if (pRasterOverlay) {
            const auto pNativeRasterOverlay = pRasterOverlay->getRasterOverlay();
            if (pNativeRasterOverlay == &rasterOverlay) {
                return rasterOverlayPath;
            }
        }
    }

    return {};
}

void OmniTileset::updateRasterOverlayAlpha(const pxr::SdfPath& rasterOverlayPath) {
    const auto rasterOverlayPaths = getRasterOverlayPaths();
    const auto rasterOverlayIndex = CppUtil::indexOf(rasterOverlayPaths, rasterOverlayPath);

    if (rasterOverlayIndex == rasterOverlayPaths.size()) {
        return;
    }

    const auto pRasterOverlay = _pContext->getAssetRegistry().getRasterOverlay(rasterOverlayPath);

    if (!pRasterOverlay) {
        return;
    }

    const auto alpha = glm::clamp(pRasterOverlay->getAlpha(), 0.0, 1.0);

    forEachFabricMaterial(_pTileset.get(), [rasterOverlayIndex, alpha](FabricMaterial& fabricMaterial) {
        fabricMaterial.setRasterOverlayAlpha(rasterOverlayIndex, alpha);
    });
}

void OmniTileset::updateDisplayColorAndOpacity() {
    const auto displayColor = getDisplayColor();
    const auto displayOpacity = getDisplayOpacity();

    forEachFabricMaterial(_pTileset.get(), [&displayColor, &displayOpacity](FabricMaterial& fabricMaterial) {
        fabricMaterial.setDisplayColorAndOpacity(displayColor, displayOpacity);
    });
}

void OmniTileset::updateShaderInput(const pxr::SdfPath& shaderPath, const pxr::TfToken& attributeName) {
    forEachFabricMaterial(_pTileset.get(), [&shaderPath, &attributeName](FabricMaterial& fabricMaterial) {
        fabricMaterial.updateShaderInput(
            FabricUtil::toFabricPath(shaderPath), FabricUtil::toFabricToken(attributeName));
    });
}

void OmniTileset::onUpdateFrame(const gsl::span<const Viewport>& viewports) {
    if (!UsdUtil::primExists(_pContext->getUsdStage(), _path)) {
        // TfNotice can be slow, and sometimes we get a frame or two before we actually get a chance to react on it.
        // This guard prevents us from crashing if the prim no longer exists.
        return;
    }

    updateTransform();
    updateView(viewports);

    if (!_extentSet) {
        _extentSet = updateExtent();
    }

    updateLoadStatus();
}

void OmniTileset::updateTransform() {
    // computeEcefToPrimWorldTransform is a slightly expensive operation to do every frame but it is simple
    // and exhaustive; it reacts to USD scene graph changes, up-axis changes, meters-per-unit changes, and georeference
    // origin changes without us needing to subscribe to any events.
    //
    // The faster approach would be to subscribe to change events for _worldPosition, _worldOrientation, _worldScale.
    // Alternatively, we could register a listener with Tf::Notice but this has the downside of only notifying us
    // about changes to the current prim and not its ancestor prims. Also Tf::Notice may notify us in a thread other
    // than the main thread and we would have to be careful to synchronize updates to Fabric in the main thread.
    const auto georeferencePath = getResolvedGeoreferencePath();
    const auto ecefToPrimWorldTransform = UsdUtil::computeEcefToPrimWorldTransform(*_pContext, georeferencePath, _path);

    // Check for transform changes and update prims accordingly
    if (ecefToPrimWorldTransform != _ecefToPrimWorldTransform) {
        _ecefToPrimWorldTransform = ecefToPrimWorldTransform;
        FabricUtil::setTilesetTransform(_pContext->getFabricStage(), _tilesetId, ecefToPrimWorldTransform);
        _extentSet = updateExtent();
    }
}

void OmniTileset::updateView(const gsl::span<const Viewport>& viewports) {
    const auto visible = UsdUtil::isPrimVisible(_pContext->getUsdStage(), _path);

    if (visible && !getSuspendUpdate()) {
        // Go ahead and select some tiles
        const auto georeferencePath = getResolvedGeoreferencePath();

        _viewStates.clear();
        for (const auto& viewport : viewports) {
            _viewStates.push_back(UsdUtil::computeViewState(*_pContext, georeferencePath, _path, viewport));
        }

        _pViewUpdateResult = &_pTileset->updateView(_viewStates);
    }

    if (!_pViewUpdateResult) {
        // No tiles have ever been selected. Return early.
        return;
    }

    // Hide tiles that we no longer need
    for (const auto pTile : _pViewUpdateResult->tilesFadingOut) {
        if (pTile->getState() == Cesium3DTilesSelection::TileLoadState::Done) {
            const auto pRenderContent = pTile->getContent().getRenderContent();
            if (pRenderContent) {
                const auto pRenderResources =
                    static_cast<const FabricRenderResources*>(pRenderContent->getRenderResources());
                if (pRenderResources) {
                    for (const auto& fabricMesh : pRenderResources->fabricMeshes) {
                        fabricMesh.pGeometry->setVisibility(false);
                    }
                }
            }
        }
    }

    // Update visibility for selected tiles
    for (const auto pTile : _pViewUpdateResult->tilesToRenderThisFrame) {
        if (pTile->getState() == Cesium3DTilesSelection::TileLoadState::Done) {
            const auto pRenderContent = pTile->getContent().getRenderContent();
            if (pRenderContent) {
                const auto pRenderResources =
                    static_cast<const FabricRenderResources*>(pRenderContent->getRenderResources());
                if (pRenderResources) {
                    for (const auto& fabricMesh : pRenderResources->fabricMeshes) {
                        fabricMesh.pGeometry->setVisibility(visible);
                    }
                }
            }
        }
    }
}

bool OmniTileset::updateExtent() {
    const auto pRootTile = _pTileset->getRootTile();
    if (!pRootTile) {
        return false;
    }

    const auto cesiumTileset = UsdUtil::getCesiumTileset(_pContext->getUsdStage(), _path);
    const auto& boundingVolume = pRootTile->getBoundingVolume();
    const auto ecefObb = Cesium3DTilesSelection::getOrientedBoundingBoxFromBoundingVolume(boundingVolume);
    const auto georeferencePath = getResolvedGeoreferencePath();
    const auto ecefToPrimWorldTransform = UsdUtil::computeEcefToPrimWorldTransform(*_pContext, georeferencePath, _path);
    const auto primObb = ecefObb.transform(ecefToPrimWorldTransform);
    const auto primAabb = primObb.toAxisAligned();

    const auto bottomLeft = glm::dvec3(primAabb.minimumX, primAabb.minimumY, primAabb.minimumZ);
    const auto topRight = glm::dvec3(primAabb.maximumX, primAabb.maximumY, primAabb.maximumZ);

    pxr::VtArray<pxr::GfVec3f> extent = {
        UsdUtil::glmToUsdVector(glm::fvec3(bottomLeft)),
        UsdUtil::glmToUsdVector(glm::fvec3(topRight)),
    };

    const auto boundable = pxr::UsdGeomBoundable(cesiumTileset);
    boundable.GetExtentAttr().Set(extent);
    return true;
}

void OmniTileset::updateLoadStatus() {
    const auto loadProgress = _pTileset->computeLoadProgress();

    if (loadProgress < 100.0f) {
        _activeLoading = true;
    } else if (_activeLoading) {
        Broadcast::tilesetLoaded(_path);
        _activeLoading = false;
    }
}

void OmniTileset::destroyNativeTileset() {
    if (_pTileset) {
        // Remove raster overlays before the native tileset is destroyed
        // See comment above _pLoadedTiles in RasterOverlayCollection.h
        while (_pTileset->getOverlays().size() > 0) {
            _pTileset->getOverlays().remove(*_pTileset->getOverlays().begin());
        }
    }

    if (_pRenderResourcesPreparer) {
        _pRenderResourcesPreparer->detachTileset();
    }

    _pTileset = nullptr;
    _pRenderResourcesPreparer = nullptr;
}

} // namespace cesium::omniverse
