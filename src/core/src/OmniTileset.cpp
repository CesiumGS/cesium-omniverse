#include "cesium/omniverse/OmniTileset.h"

#include "cesium/omniverse/Broadcast.h"
#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/FabricGeometry.h"
#include "cesium/omniverse/FabricMaterial.h"
#include "cesium/omniverse/FabricPrepareRenderResources.h"
#include "cesium/omniverse/FabricUtil.h"
#include "cesium/omniverse/GeospatialUtil.h"
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
#include <CesiumUsdSchemas/tileset.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/boundable.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>

namespace cesium::omniverse {

OmniTileset::OmniTileset(const pxr::SdfPath& tilesetPath, const pxr::SdfPath& georeferencePath)
    : _tilesetPath(tilesetPath)
    , _tilesetId(Context::instance().getNextTilesetId()) {
    reload();

    UsdUtil::setGeoreferenceForTileset(tilesetPath, georeferencePath);
}

OmniTileset::~OmniTileset() {
    _renderResourcesPreparer->detachTileset();
}

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

float OmniTileset::getMainThreadLoadingTimeLimit() const {
    auto tileset = UsdUtil::getCesiumTileset(_tilesetPath);

    float mainThreadLoadingTimeLimit;
    tileset.GetMainThreadLoadingTimeLimitAttr().Get<float>(&mainThreadLoadingTimeLimit);

    return mainThreadLoadingTimeLimit;
}

pxr::CesiumGeoreference OmniTileset::getGeoreference() const {
    auto tileset = UsdUtil::getCesiumTileset(_tilesetPath);

    pxr::SdfPathVector targets;
    tileset.GetGeoreferenceBindingRel().GetTargets(&targets);
    assert(!targets.empty());

    // We only care about the first target.
    const auto georeferencePath = targets[0];
    return UsdUtil::getCesiumGeoreference(georeferencePath);
}

pxr::SdfPath OmniTileset::getMaterialPath() const {
    auto tileset = UsdUtil::getCesiumTileset(_tilesetPath);

    const auto materialBindingApi = pxr::UsdShadeMaterialBindingAPI(tileset);
    const auto materialBinding = materialBindingApi.GetDirectBinding();
    const auto& materialPath = materialBinding.GetMaterialPath();

    return materialPath;
}

int64_t OmniTileset::getTilesetId() const {
    return _tilesetId;
}

TilesetStatistics OmniTileset::getStatistics() const {
    TilesetStatistics statistics;

    statistics.tilesetCachedBytes = static_cast<uint64_t>(_tileset->getTotalDataBytes());
    statistics.tilesLoaded = static_cast<uint64_t>(_tileset->getNumberOfTilesLoaded());

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

void OmniTileset::reload() {
    if (_renderResourcesPreparer != nullptr) {
        _renderResourcesPreparer->detachTileset();
    }

    _renderResourcesPreparer = std::make_shared<FabricPrepareRenderResources>(*this);
    auto& context = Context::instance();
    auto asyncSystem = CesiumAsync::AsyncSystem(context.getTaskProcessor());
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
    options.mainThreadLoadingTimeLimit = getMainThreadLoadingTimeLimit();

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

    CesiumGltf::SupportedGpuCompressedPixelFormats supportedFormats;

    // Only BCN compressed texture formats are supported in Omniverse
    supportedFormats.ETC1_RGB = false;
    supportedFormats.ETC2_RGBA = false;
    supportedFormats.BC1_RGB = true;
    supportedFormats.BC3_RGBA = true;
    supportedFormats.BC4_R = true;
    supportedFormats.BC5_RG = true;
    supportedFormats.BC7_RGBA = true;
    supportedFormats.PVRTC1_4_RGB = false;
    supportedFormats.PVRTC1_4_RGBA = false;
    supportedFormats.ASTC_4x4_RGBA = false;
    supportedFormats.PVRTC2_4_RGB = false;
    supportedFormats.PVRTC2_4_RGBA = false;
    supportedFormats.ETC2_EAC_R11 = false;
    supportedFormats.ETC2_EAC_RG11 = false;

    options.contentOptions.ktx2TranscodeTargets = CesiumGltf::Ktx2TranscodeTargets(supportedFormats, false);

    _pViewUpdateResult = nullptr;
    _extentSet = false;
    _activeLoading = false;
    _imageryPaths.clear();

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
    options.showCreditsOnScreen = imagery.getShowCreditsOnScreen();

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
    _imageryPaths.push_back(imageryPath);
}

std::optional<uint64_t> OmniTileset::findImageryLayerIndex(const Cesium3DTilesSelection::RasterOverlay& overlay) const {
    uint64_t imageryLayerIndex = 0;
    for (const auto& pOverlay : _tileset->getOverlays()) {
        if (&overlay == pOverlay.get()) {
            return imageryLayerIndex;
        }

        imageryLayerIndex++;
    }

    return std::nullopt;
}

std::optional<uint64_t> OmniTileset::findImageryLayerIndex(const pxr::SdfPath& imageryPath) const {
    uint64_t imageryLayerIndex = 0;
    for (const auto& _imageryPath : _imageryPaths) {
        if (_imageryPath == imageryPath) {
            return imageryLayerIndex;
        }

        imageryLayerIndex++;
    }

    return std::nullopt;
}

uint64_t OmniTileset::getImageryLayerCount() const {
    return _tileset->getOverlays().size();
}

void OmniTileset::updateImageryAlpha(const pxr::SdfPath& imageryPath) {
    const auto imageryLayerIndex = findImageryLayerIndex(imageryPath);
    if (!imageryLayerIndex.has_value()) {
        return;
    }

    auto alpha = OmniImagery(imageryPath).getAlpha();
    alpha = glm::clamp(alpha, 0.0f, 1.0f);

    _tileset->forEachLoadedTile([imageryLayerIndex, alpha](Cesium3DTilesSelection::Tile& tile) {
        if (tile.getState() != Cesium3DTilesSelection::TileLoadState::Done) {
            return;
        }
        const auto& content = tile.getContent();
        const auto pRenderContent = content.getRenderContent();
        if (!pRenderContent) {
            return;
        }
        const auto pTileRenderResources = static_cast<TileRenderResources*>(pRenderContent->getRenderResources());
        if (!pTileRenderResources) {
            return;
        }
        for (const auto& fabricMesh : pTileRenderResources->fabricMeshes) {
            if (!fabricMesh.material) {
                continue;
            }
            fabricMesh.material->setImageryLayerAlpha(imageryLayerIndex.value(), alpha);
        }
    });
}

void OmniTileset::onUpdateFrame(const std::vector<Viewport>& viewports) {
    if (!UsdUtil::primExists(_tilesetPath)) {
        // TfNotice can be slow, and sometimes we get a frame or two before we actually get a chance to react on it.
        //   This guard prevents us from crashing if the prim no longer exists.
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
    // computeEcefToUsdTransformForPrim is slightly expensive operations to do every frame but it is simple
    // and exhaustive. E.g. it reacts to USD scene graph changes, up-axis changes, meters-per-unit changes, and georeference origin changes
    // without us needing to subscribe to any events.
    //
    // The faster approach would be to load the tileset USD prim into Fabric (via usdrt::UsdStage::GetPrimAtPath)
    // and subscribe to change events for _worldPosition, _worldOrientation, _worldScale.
    // Alternatively, we could register a listener with Tf::Notice but this has the downside of only notifying us
    // about changes to the current prim and not its ancestor prims. Also Tf::Notice may notify us in a thread other
    // than the main thread and we would have to be careful to synchronize updates to Fabric in the main thread.

    const auto georeferenceOrigin = GeospatialUtil::convertGeoreferenceToCartographic(getGeoreference());
    const auto ecefToUsdTransform = UsdUtil::computeEcefToUsdWorldTransformForPrim(georeferenceOrigin, _tilesetPath);

    // Check for transform changes and update prims accordingly
    if (ecefToUsdTransform != _ecefToUsdTransform) {
        _ecefToUsdTransform = ecefToUsdTransform;
        FabricUtil::setTilesetTransform(_tilesetId, ecefToUsdTransform);
        updateExtent();
    }
}

void OmniTileset::updateView(const std::vector<Viewport>& viewports) {
    const auto visible = UsdUtil::isPrimVisible(_tilesetPath);

    if (visible && !getSuspendUpdate()) {
        // Go ahead and select some tiles
        const auto& georeferenceOrigin = GeospatialUtil::convertGeoreferenceToCartographic(getGeoreference());

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

    // Hide tiles that we no longer need
    for (const auto tile : _pViewUpdateResult->tilesFadingOut) {
        if (tile->getState() == Cesium3DTilesSelection::TileLoadState::Done) {
            const auto pRenderContent = tile->getContent().getRenderContent();
            if (pRenderContent) {
                const auto pRenderResources = pRenderContent->getRenderResources();
                if (pRenderResources) {
                    const auto pTileRenderResources = reinterpret_cast<TileRenderResources*>(pRenderResources);
                    for (const auto& fabricMesh : pTileRenderResources->fabricMeshes) {
                        fabricMesh.geometry->setVisibility(false);
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
                        fabricMesh.geometry->setVisibility(visible);
                    }
                }
            }
        }
    }
}

bool OmniTileset::updateExtent() {
    auto rootTile = _tileset->getRootTile();
    if (rootTile == nullptr) {
        return false;
    }

    const auto tileset = UsdUtil::getCesiumTileset(_tilesetPath);
    const auto& bounding_volume = rootTile->getBoundingVolume();
    const auto oriented = Cesium3DTilesSelection::getOrientedBoundingBoxFromBoundingVolume(bounding_volume);
    const auto georeferenceOrigin = Context::instance().getGeoreferenceOrigin();
    const auto ecefToUsdTransform = UsdUtil::computeEcefToUsdWorldTransformForPrim(georeferenceOrigin, _tilesetPath);
    const auto usdOriented = oriented.transform(ecefToUsdTransform);
    const auto& center = usdOriented.getCenter();

    const auto& halfAxes = usdOriented.getHalfAxes();
    const auto xLengthHalf = static_cast<float>(glm::length(halfAxes[0]));
    const auto yLengthHalf = static_cast<float>(glm::length(halfAxes[1]));
    const auto zLengthHalf = static_cast<float>(glm::length(halfAxes[2]));

    pxr::VtArray<pxr::GfVec3f> extent;
    const auto centerGf =
        pxr::GfVec3f(static_cast<float>(center.x), static_cast<float>(center.y), static_cast<float>(center.z));
    extent.push_back(pxr::GfVec3f(-xLengthHalf, -yLengthHalf, -zLengthHalf) + centerGf);
    extent.push_back(pxr::GfVec3f(xLengthHalf, yLengthHalf, zLengthHalf) + centerGf);

    auto boundable = pxr::UsdGeomBoundable(tileset);
    boundable.GetExtentAttr().Set(extent);
    return true;
}

void OmniTileset::updateLoadStatus() {
    const auto loadProgress = _tileset->computeLoadProgress();

    if (loadProgress < 100.0f) {
        _activeLoading = true;
    } else if (_activeLoading) {
        Broadcast::tilesetLoaded(_tilesetPath);
        _activeLoading = false;
    }
}

} // namespace cesium::omniverse
