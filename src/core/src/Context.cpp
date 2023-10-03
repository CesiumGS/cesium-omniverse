#include "cesium/omniverse/Context.h"

#include "cesium/omniverse/AssetRegistry.h"
#include "cesium/omniverse/Broadcast.h"
#include "cesium/omniverse/CesiumIonSession.h"
#include "cesium/omniverse/FabricResourceManager.h"
#include "cesium/omniverse/FabricUtil.h"
#include "cesium/omniverse/GeospatialUtil.h"
#include "cesium/omniverse/GlobeAnchorRegistry.h"
#include "cesium/omniverse/HttpAssetAccessor.h"
#include "cesium/omniverse/LoggerSink.h"
#include "cesium/omniverse/OmniImagery.h"
#include "cesium/omniverse/OmniTileset.h"
#include "cesium/omniverse/TaskProcessor.h"
#include "cesium/omniverse/Tokens.h"
#include "cesium/omniverse/UsdUtil.h"

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <Cesium3DTilesSelection/CreditSystem.h>
#include <Cesium3DTilesSelection/Tileset.h>
#include <Cesium3DTilesSelection/registerAllTileContentTypes.h>
#include <CesiumUsdSchemas/data.h>
#include <CesiumUsdSchemas/imagery.h>
#include <CesiumUsdSchemas/tileset.h>
#include <CesiumUsdSchemas/tokens.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdUtils/stageCache.h>

#if CESIUM_TRACING_ENABLED
#include <chrono>
#endif

namespace cesium::omniverse {

namespace {

std::unique_ptr<Context> context;

} // namespace

void Context::onStartup(const std::filesystem::path& cesiumExtensionLocation) {
    static int64_t contextId = 0;

    // Shut down the current context (if it exists)
    onShutdown();

    // Create a new context
    context = std::make_unique<Context>();

    // Initialize the context. This needs to happen after the global variable is set
    // since code inside initialize may call Context::instance.
    context->initialize(contextId++, cesiumExtensionLocation);
}

void Context::onShutdown() {
    if (context) {
        // Destroy the context. This needs to happen before the global variable is
        // reset since code inside destroy may call Context::instance.
        context->destroy();
        context.reset();
    }
}

Context& Context::instance() {
    return *context.get();
}

void Context::initialize(int64_t contextId, const std::filesystem::path& cesiumExtensionLocation) {
    _contextId = contextId;
    _tilesetId = 0;

    _cesiumExtensionLocation = cesiumExtensionLocation.lexically_normal();
    _certificatePath = _cesiumExtensionLocation / "certs" / "cacert.pem";
    const auto cesiumMdlPath = _cesiumExtensionLocation / "mdl" / "cesium.mdl";
    _cesiumMdlPathToken = pxr::TfToken(cesiumMdlPath.generic_string());

    _taskProcessor = std::make_shared<TaskProcessor>();
    _httpAssetAccessor = std::make_shared<HttpAssetAccessor>(_certificatePath);
    _creditSystem = std::make_shared<Cesium3DTilesSelection::CreditSystem>();

    _logger = std::make_shared<spdlog::logger>(
        std::string("cesium-omniverse"),
        spdlog::sinks_init_list{
            std::make_shared<LoggerSink>(omni::log::Level::eVerbose),
            std::make_shared<LoggerSink>(omni::log::Level::eInfo),
            std::make_shared<LoggerSink>(omni::log::Level::eWarn),
            std::make_shared<LoggerSink>(omni::log::Level::eError),
            std::make_shared<LoggerSink>(omni::log::Level::eFatal),
        });

    CesiumAsync::AsyncSystem asyncSystem{_taskProcessor};
    _session = std::make_shared<CesiumIonSession>(asyncSystem, _httpAssetAccessor);
    _session->resume();

    Cesium3DTilesSelection::registerAllTileContentTypes();

#if CESIUM_TRACING_ENABLED
    const auto timeNow = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::steady_clock::now());
    const auto timeSinceEpoch = timeNow.time_since_epoch().count();
    const auto path = cesiumExtensionLocation / fmt::format("cesium-trace-{}.json", timeSinceEpoch);
    CESIUM_TRACE_INIT(path.string());
#endif
}

void Context::destroy() {
    clearStage();

    CESIUM_TRACE_SHUTDOWN();
}

std::shared_ptr<TaskProcessor> Context::getTaskProcessor() {
    return _taskProcessor;
}

std::shared_ptr<HttpAssetAccessor> Context::getHttpAssetAccessor() {
    return _httpAssetAccessor;
}

std::shared_ptr<Cesium3DTilesSelection::CreditSystem> Context::getCreditSystem() {
    return _creditSystem;
}

std::shared_ptr<spdlog::logger> Context::getLogger() {
    return _logger;
}

void Context::setProjectDefaultToken(const CesiumIonClient::Token& token) {
    if (token.token.empty()) {
        return;
    }

    const auto cesiumDataUsd = UsdUtil::getOrCreateCesiumData();

    cesiumDataUsd.GetProjectDefaultIonAccessTokenAttr().Set<std::string>(token.token);
    cesiumDataUsd.GetProjectDefaultIonAccessTokenIdAttr().Set<std::string>(token.id);
}

void Context::reloadTileset(const pxr::SdfPath& tilesetPath) {
    const auto tileset = AssetRegistry::getInstance().getTilesetByPath(tilesetPath);

    if (!tileset.has_value()) {
        return;
    }

    tileset.value()->reload();
}

void Context::clearStage() {
    // The order is important. Clear tilesets first so that Fabric resources are released back into the pool. Then clear the pools.
    AssetRegistry::getInstance().clear();
    FabricResourceManager::getInstance().clear();
    GlobeAnchorRegistry::getInstance().clear();
}

void Context::reloadStage() {
    clearStage();

    auto& fabricResourceManager = FabricResourceManager::getInstance();
    fabricResourceManager.setDisableMaterials(getDebugDisableMaterials());
    fabricResourceManager.setDisableTextures(getDebugDisableTextures());
    fabricResourceManager.setDisableGeometryPool(getDebugDisableGeometryPool());
    fabricResourceManager.setDisableMaterialPool(getDebugDisableMaterialPool());
    fabricResourceManager.setDisableTexturePool(getDebugDisableTexturePool());
    fabricResourceManager.setGeometryPoolInitialCapacity(getDebugGeometryPoolInitialCapacity());
    fabricResourceManager.setMaterialPoolInitialCapacity(getDebugMaterialPoolInitialCapacity());
    fabricResourceManager.setTexturePoolInitialCapacity(getDebugTexturePoolInitialCapacity());
    fabricResourceManager.setDebugRandomColors(getDebugRandomColors());

    // Repopulate the asset registry. We need to do this manually because USD doesn't notify us about
    // resynced paths when the stage is loaded.
    const auto stage = UsdUtil::getUsdStage();
    for (const auto& prim : stage->Traverse()) {
        const auto& path = prim.GetPath();
        if (UsdUtil::isCesiumTileset(path)) {
            AssetRegistry::getInstance().addTileset(path, UsdUtil::GEOREFERENCE_PATH);
        } else if (UsdUtil::isCesiumImagery(path)) {
            AssetRegistry::getInstance().addImagery(path);
        }
    }
}

void Context::onUpdateFrame(const std::vector<Viewport>& viewports) {
    processUsdNotifications();

    const auto georeferenceOrigin = Context::instance().getGeoreferenceOrigin();
    const auto ecefToUsdTransform = UsdUtil::computeEcefToUsdTransform(georeferenceOrigin);

    // Check if the ecefToUsd transform has changed and update CesiumSession
    if (ecefToUsdTransform != _ecefToUsdTransform) {
        _ecefToUsdTransform = ecefToUsdTransform;

        const auto stage = UsdUtil::getUsdStage();
        const UsdUtil::ScopedEdit scopedEdit(stage);
        auto cesiumSession = UsdUtil::getOrCreateCesiumSession();
        cesiumSession.GetEcefToUsdTransformAttr().Set(pxr::VtValue(UsdUtil::glmToUsdMatrix(ecefToUsdTransform)));
    }

    const auto& tilesets = AssetRegistry::getInstance().getAllTilesets();
    for (const auto& tileset : tilesets) {
        tileset->onUpdateFrame(viewports);
    }
}

void Context::processPropertyChanged(const ChangedPrim& changedPrim) {
    const auto& [path, name, primType, changeType] = changedPrim;

    switch (primType) {
        case ChangedPrimType::CESIUM_DATA:
            return processCesiumDataChanged(changedPrim);
        case ChangedPrimType::CESIUM_TILESET:
            return processCesiumTilesetChanged(changedPrim);
        case ChangedPrimType::CESIUM_IMAGERY:
            return processCesiumImageryChanged(changedPrim);
        case ChangedPrimType::CESIUM_GEOREFERENCE:
            return processCesiumGeoreferenceChanged(changedPrim);
        case ChangedPrimType::CESIUM_GLOBE_ANCHOR:
            return processCesiumGlobeAnchorChanged(changedPrim);
        default:
            return;
    }
}

void Context::processCesiumDataChanged(const ChangedPrim& changedPrim) {
    const auto& [path, name, primType, changeType] = changedPrim;

    if (name == pxr::CesiumTokens->cesiumProjectDefaultIonAccessToken) {
        // Reload tilesets that use the project default token
        const auto& tilesets = AssetRegistry::getInstance().getAllTilesets();
        for (const auto& tileset : tilesets) {
            const auto tilesetToken = tileset->getIonAccessToken();
            const auto defaultToken = Context::instance().getDefaultToken();
            if (!tilesetToken.has_value() ||
                (defaultToken.has_value() && tilesetToken.value().token == defaultToken.value().token)) {
                tileset->reload();
            }
        }
    } else if (
        name == pxr::CesiumTokens->cesiumDebugDisableMaterials ||
        name == pxr::CesiumTokens->cesiumDebugDisableTextures ||
        name == pxr::CesiumTokens->cesiumDebugDisableGeometryPool ||
        name == pxr::CesiumTokens->cesiumDebugDisableMaterialPool ||
        name == pxr::CesiumTokens->cesiumDebugGeometryPoolInitialCapacity ||
        name == pxr::CesiumTokens->cesiumDebugMaterialPoolInitialCapacity ||
        name == pxr::CesiumTokens->cesiumDebugTexturePoolInitialCapacity ||
        name == pxr::CesiumTokens->cesiumDebugRandomColors) {
        reloadStage();
    }
}

void Context::processCesiumTilesetChanged(const ChangedPrim& changedPrim) {
    const auto& [path, name, primType, changeType] = changedPrim;

    const auto tileset = AssetRegistry::getInstance().getTilesetByPath(path);
    if (!tileset.has_value()) {
        return;
    }

    // clang-format off
    if (name == pxr::CesiumTokens->cesiumSourceType ||
        name == pxr::CesiumTokens->cesiumUrl ||
        name == pxr::CesiumTokens->cesiumIonAssetId ||
        name == pxr::CesiumTokens->cesiumIonAccessToken ||
        name == pxr::CesiumTokens->cesiumMaximumScreenSpaceError ||
        name == pxr::CesiumTokens->cesiumPreloadAncestors ||
        name == pxr::CesiumTokens->cesiumPreloadSiblings ||
        name == pxr::CesiumTokens->cesiumForbidHoles ||
        name == pxr::CesiumTokens->cesiumMaximumSimultaneousTileLoads ||
        name == pxr::CesiumTokens->cesiumMaximumCachedBytes ||
        name == pxr::CesiumTokens->cesiumLoadingDescendantLimit ||
        name == pxr::CesiumTokens->cesiumEnableFrustumCulling ||
        name == pxr::CesiumTokens->cesiumEnableFogCulling ||
        name == pxr::CesiumTokens->cesiumEnforceCulledScreenSpaceError ||
        name == pxr::CesiumTokens->cesiumCulledScreenSpaceError ||
        name == pxr::CesiumTokens->cesiumSmoothNormals ||
        name == pxr::CesiumTokens->cesiumMainThreadLoadingTimeLimit ||
        name == pxr::CesiumTokens->cesiumShowCreditsOnScreen ||
        name == pxr::UsdTokens->material_binding) {
        tileset.value()->reload();
    }
    // clang-format on
}

void Context::processCesiumImageryChanged(const ChangedPrim& changedPrim) {
    const auto& [path, name, primType, changeType] = changedPrim;

    const auto tilesetPath = path.GetParentPath();
    const auto tileset = AssetRegistry::getInstance().getTilesetByPath(tilesetPath);
    if (!tileset.has_value()) {
        return;
    }

    // clang-format off
    if (name == pxr::CesiumTokens->cesiumIonAssetId ||
        name == pxr::CesiumTokens->cesiumIonAccessToken ||
        name == pxr::CesiumTokens->cesiumShowCreditsOnScreen) {
        // Reload the tileset that the imagery is attached to
        tileset.value()->reload();
    }
    // clang-format on
}

void Context::processCesiumGeoreferenceChanged(const cesium::omniverse::ChangedPrim& changedPrim) {
    const auto& [path, name, primType, changeType] = changedPrim;

    auto anchors = GlobeAnchorRegistry::getInstance().getAllAnchors();
    for (const auto& globeAnchor : anchors) {
        auto anchorApi = UsdUtil::getCesiumGlobeAnchor(globeAnchor->getPrimPath());

        pxr::SdfPathVector targets;
        if (!anchorApi.GetGeoreferenceBindingRel().GetForwardedTargets(&targets)) {
            return;
        }

        // We only want to update an anchor if we are updating it's related Georeference Prim.
        if (path != targets[0]) {
            return;
        }

        auto georeferenceOrigin = UsdUtil::getCesiumGeoreference(targets[0]);
        auto origin = GeospatialUtil::convertGeoreferenceToCartographic(georeferenceOrigin);

        GeospatialUtil::updateAnchorOrigin(origin, anchorApi, globeAnchor);
    }
}

void Context::processCesiumGlobeAnchorChanged(const cesium::omniverse::ChangedPrim& changedPrim) {
    const auto& [path, name, primType, changeType] = changedPrim;

    auto globeAnchor = UsdUtil::getCesiumGlobeAnchor(path);
    pxr::SdfPathVector targets;
    if (!globeAnchor.GetGeoreferenceBindingRel().GetForwardedTargets(&targets)) {
        return;
    }
    auto georeferenceOrigin = UsdUtil::getCesiumGeoreference(targets[0]);
    auto cartographicOrigin = GeospatialUtil::convertGeoreferenceToCartographic(georeferenceOrigin);

    bool detectTransformChanges;
    globeAnchor.GetDetectTransformChangesAttr().Get(&detectTransformChanges);

    if (detectTransformChanges && (name == pxr::CesiumTokens->cesiumAnchorDetectTransformChanges ||
                                   name == pxr::UsdTokens->xformOp_transform_cesium)) {
        GeospatialUtil::updateAnchorByUsdTransform(cartographicOrigin, globeAnchor);

        return;
    }

    if (name == pxr::CesiumTokens->cesiumAnchorGeographicCoordinates) {
        GeospatialUtil::updateAnchorByLatLongHeight(cartographicOrigin, globeAnchor);

        return;
    }

    if (name == pxr::CesiumTokens->cesiumAnchorPosition || name == pxr::CesiumTokens->cesiumAnchorRotation ||
        name == pxr::CesiumTokens->cesiumAnchorScale) {
        GeospatialUtil::updateAnchorByFixedTransform(cartographicOrigin, globeAnchor);

        return;
    }
}

void Context::processPrimRemoved(const ChangedPrim& changedPrim) {
    // TODO: Remove prim from anchor registry if has anchor API.

    if (changedPrim.primType == ChangedPrimType::CESIUM_TILESET) {
        // Remove the tileset from the asset registry
        const auto tilesetPath = changedPrim.path;
        AssetRegistry::getInstance().removeTileset(tilesetPath);
    } else if (changedPrim.primType == ChangedPrimType::CESIUM_IMAGERY) {
        // Remove the imagery from the asset registry and reload the tileset that the imagery was attached to
        const auto imageryPath = changedPrim.path;
        const auto tilesetPath = changedPrim.path.GetParentPath();
        AssetRegistry::getInstance().removeImagery(imageryPath);
        reloadTileset(tilesetPath);
    }
}

void Context::processPrimAdded(const ChangedPrim& changedPrim) {
    if (changedPrim.primType == ChangedPrimType::CESIUM_TILESET) {
        // Add the tileset to the asset registry
        const auto tilesetPath = changedPrim.path;
        AssetRegistry::getInstance().addTileset(tilesetPath, UsdUtil::GEOREFERENCE_PATH);
    } else if (changedPrim.primType == ChangedPrimType::CESIUM_IMAGERY) {
        // Add the imagery to the asset registry and reload the tileset that the imagery is attached to
        const auto imageryPath = changedPrim.path;
        const auto tilesetPath = changedPrim.path.GetParentPath();
        AssetRegistry::getInstance().addImagery(imageryPath);
        reloadTileset(tilesetPath);
    }
}

void Context::processUsdNotifications() {
    const auto changedPrims = _usdNotificationHandler.popChangedPrims();

    for (const auto& change : changedPrims) {
        switch (change.changeType) {
            case ChangeType::PROPERTY_CHANGED:
                processPropertyChanged(change);
                break;
            case ChangeType::PRIM_REMOVED:
                processPrimRemoved(change);
                break;
            case ChangeType::PRIM_ADDED:
                processPrimAdded(change);
                break;
            default:
                break;
        }
    }
}

void Context::onUpdateUi() {
    if (_session == nullptr) {
        return;
    }

    _session->tick();
}

pxr::UsdStageRefPtr Context::getStage() const {
    return _stage;
}

omni::fabric::StageReaderWriter Context::getFabricStageReaderWriter() const {
    assert(_fabricStageReaderWriter.has_value());
    return _fabricStageReaderWriter.value(); // NOLINT(bugprone-unchecked-optional-access)
}

long Context::getStageId() const {
    return _stageId;
}

void Context::setStageId(long stageId) {
    const auto oldStage = _stageId;
    const auto newStage = stageId;

    if (oldStage == newStage) {
        // No change
        return;
    }

    if (oldStage > 0) {
        // Remove references to the old stage
        _stage.Reset();
        _fabricStageReaderWriter.reset();
        _stageId = 0;

        // Now it's safe to clear anything else that references the stage
        clearStage();
    }

    if (newStage > 0) {
        // Set the USD stage
        _stage = pxr::UsdUtilsStageCache::Get().Find(pxr::UsdStageCache::Id::FromLongInt(stageId));

        // Set the Fabric stage
        const auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
        const auto stageReaderWriterId =
            iStageReaderWriter->get(omni::fabric::UsdStageId{static_cast<uint64_t>(stageId)});
        _fabricStageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);

        // Ensure that the CesiumData prim exists so that we can set the georeference
        // and other top-level properties without waiting for an ion session to start
        UsdUtil::getOrCreateCesiumData();
        UsdUtil::getOrCreateCesiumGeoreference();
        UsdUtil::getOrCreateCesiumSession();

        // Repopulate the asset registry
        reloadStage();
    }

    _stageId = stageId;
}

int64_t Context::getContextId() const {
    return _contextId;
}

int64_t Context::getNextTilesetId() const {
    return _tilesetId++;
}

const CesiumGeospatial::Cartographic Context::getGeoreferenceOrigin() const {
    const auto georeference = UsdUtil::getOrCreateCesiumGeoreference();

    return GeospatialUtil::convertGeoreferenceToCartographic(georeference);
}

void Context::setGeoreferenceOrigin(const CesiumGeospatial::Cartographic& origin) {
    const auto georeference = UsdUtil::getOrCreateCesiumGeoreference();

    georeference.GetGeoreferenceOriginLongitudeAttr().Set<double>(glm::degrees(origin.longitude));
    georeference.GetGeoreferenceOriginLatitudeAttr().Set<double>(glm::degrees(origin.latitude));
    georeference.GetGeoreferenceOriginHeightAttr().Set<double>(origin.height);
}

void Context::connectToIon() {
    if (_session == nullptr) {
        return;
    }

    _session->connect();
}

std::optional<std::shared_ptr<CesiumIonSession>> Context::getSession() {
    if (_session == nullptr) {
        return std::nullopt;
    }

    return std::optional<std::shared_ptr<CesiumIonSession>>{_session};
}

std::optional<CesiumIonClient::Token> Context::getDefaultToken() const {
    const auto cesiumDataUsd = UsdUtil::getOrCreateCesiumData();

    std::string projectDefaultToken;
    std::string projectDefaultTokenId;

    cesiumDataUsd.GetProjectDefaultIonAccessTokenAttr().Get(&projectDefaultToken);
    cesiumDataUsd.GetProjectDefaultIonAccessTokenIdAttr().Get(&projectDefaultTokenId);

    if (projectDefaultToken.empty()) {
        return std::nullopt;
    }

    return CesiumIonClient::Token{projectDefaultTokenId, "", projectDefaultToken};
}

SetDefaultTokenResult Context::getSetDefaultTokenResult() const {
    return _lastSetTokenResult;
}

bool Context::isDefaultTokenSet() const {
    return getDefaultToken().has_value();
}

void Context::createToken(const std::string& name) {
    auto connection = _session->getConnection();

    if (!connection.has_value()) {
        _lastSetTokenResult = SetDefaultTokenResult{
            SetDefaultTokenResultCode::NOT_CONNECTED_TO_ION,
            SetDefaultTokenResultMessages::NOT_CONNECTED_TO_ION_MESSAGE};
        return;
    }

    connection->createToken(name, {"assets:read"}, std::vector<int64_t>{1}, std::nullopt)
        .thenInMainThread([this](CesiumIonClient::Response<CesiumIonClient::Token>&& response) {
            if (response.value) {
                setProjectDefaultToken(response.value.value());

                _lastSetTokenResult =
                    SetDefaultTokenResult{SetDefaultTokenResultCode::OK, SetDefaultTokenResultMessages::OK_MESSAGE};
            } else {
                _lastSetTokenResult = SetDefaultTokenResult{
                    SetDefaultTokenResultCode::CREATE_FAILED,
                    fmt::format(
                        SetDefaultTokenResultMessages::CREATE_FAILED_MESSAGE_BASE,
                        response.errorMessage,
                        response.errorCode)};
            }

            Broadcast::setDefaultTokenComplete();
        });
}
void Context::selectToken(const CesiumIonClient::Token& token) {
    auto connection = _session->getConnection();

    if (!connection.has_value()) {
        _lastSetTokenResult = SetDefaultTokenResult{
            SetDefaultTokenResultCode::NOT_CONNECTED_TO_ION,
            SetDefaultTokenResultMessages::NOT_CONNECTED_TO_ION_MESSAGE};
    } else {
        setProjectDefaultToken(token);

        _lastSetTokenResult =
            SetDefaultTokenResult{SetDefaultTokenResultCode::OK, SetDefaultTokenResultMessages::OK_MESSAGE};
    }

    Broadcast::setDefaultTokenComplete();
}
void Context::specifyToken(const std::string& token) {
    _session->findToken(token).thenInMainThread(
        [this, token](CesiumIonClient::Response<CesiumIonClient::Token>&& response) {
            if (response.value) {
                setProjectDefaultToken(response.value.value());
            } else {
                CesiumIonClient::Token t;
                t.token = token;
                setProjectDefaultToken(t);
            }
            // We assume the user knows what they're doing if they specify a token not on their account.
            _lastSetTokenResult =
                SetDefaultTokenResult{SetDefaultTokenResultCode::OK, SetDefaultTokenResultMessages::OK_MESSAGE};

            Broadcast::setDefaultTokenComplete();
        });
}

std::optional<AssetTroubleshootingDetails> Context::getAssetTroubleshootingDetails() {
    return _assetTroubleshootingDetails;
}
std::optional<TokenTroubleshootingDetails> Context::getAssetTokenTroubleshootingDetails() {
    return _assetTokenTroubleshootingDetails;
}
std::optional<TokenTroubleshootingDetails> Context::getDefaultTokenTroubleshootingDetails() {
    return _defaultTokenTroubleshootingDetails;
}
void Context::updateTroubleshootingDetails(
    const pxr::SdfPath& tilesetPath,
    int64_t tilesetIonAssetId,
    uint64_t tokenEventId,
    uint64_t assetEventId) {
    const auto tileset = AssetRegistry::getInstance().getTilesetByPath(tilesetPath);

    if (!tileset.has_value()) {
        return;
    }

    TokenTroubleshooter troubleshooter;

    _assetTroubleshootingDetails = AssetTroubleshootingDetails();
    troubleshooter.updateAssetTroubleshootingDetails(
        tilesetIonAssetId, assetEventId, _assetTroubleshootingDetails.value());

    _defaultTokenTroubleshootingDetails = TokenTroubleshootingDetails();

    const auto& defaultToken = getDefaultToken();
    if (defaultToken.has_value()) {
        const auto& token = defaultToken.value().token;
        troubleshooter.updateTokenTroubleshootingDetails(
            tilesetIonAssetId, token, tokenEventId, _defaultTokenTroubleshootingDetails.value());
    }

    _assetTokenTroubleshootingDetails = TokenTroubleshootingDetails();

    auto tilesetIonAccessToken = tileset.value()->getIonAccessToken();
    if (tilesetIonAccessToken.has_value()) {
        troubleshooter.updateTokenTroubleshootingDetails(
            tilesetIonAssetId,
            tilesetIonAccessToken.value().token,
            tokenEventId,
            _assetTokenTroubleshootingDetails.value());
    }
}
void Context::updateTroubleshootingDetails(
    const pxr::SdfPath& tilesetPath,
    [[maybe_unused]] int64_t tilesetIonAssetId,
    int64_t imageryIonAssetId,
    uint64_t tokenEventId,
    uint64_t assetEventId) {
    auto& registry = AssetRegistry::getInstance();
    const auto tileset = registry.getTilesetByPath(tilesetPath);
    if (!tileset.has_value()) {
        return;
    }

    const auto imagery = registry.getImageryByIonAssetId(imageryIonAssetId);
    if (!imagery.has_value()) {
        return;
    }

    TokenTroubleshooter troubleshooter;

    _assetTroubleshootingDetails = AssetTroubleshootingDetails();
    troubleshooter.updateAssetTroubleshootingDetails(
        imageryIonAssetId, assetEventId, _assetTroubleshootingDetails.value());

    _defaultTokenTroubleshootingDetails = TokenTroubleshootingDetails();

    const auto& defaultToken = getDefaultToken();
    if (defaultToken.has_value()) {
        const auto& token = defaultToken.value().token;
        troubleshooter.updateTokenTroubleshootingDetails(
            imageryIonAssetId, token, tokenEventId, _defaultTokenTroubleshootingDetails.value());
    }

    _assetTokenTroubleshootingDetails = TokenTroubleshootingDetails();

    auto imageryIonAccessToken = imagery.value()->getIonAccessToken();
    if (imageryIonAccessToken.has_value()) {
        troubleshooter.updateTokenTroubleshootingDetails(
            imageryIonAssetId,
            imageryIonAccessToken.value().token,
            tokenEventId,
            _assetTokenTroubleshootingDetails.value());
    }
}

const std::filesystem::path& Context::getCesiumExtensionLocation() const {
    return _cesiumExtensionLocation;
}

const std::filesystem::path& Context::getCertificatePath() const {
    return _certificatePath;
}

const pxr::TfToken& Context::getCesiumMdlPathToken() const {
    return _cesiumMdlPathToken;
}

bool Context::getDebugDisableMaterials() const {
    const auto cesiumDataUsd = UsdUtil::getOrCreateCesiumData();
    bool disableMaterials;
    cesiumDataUsd.GetDebugDisableMaterialsAttr().Get(&disableMaterials);
    return disableMaterials;
}

bool Context::getDebugDisableTextures() const {
    const auto cesiumDataUsd = UsdUtil::getOrCreateCesiumData();
    bool disableTextures;
    cesiumDataUsd.GetDebugDisableTexturesAttr().Get(&disableTextures);
    return disableTextures;
}

bool Context::getDebugDisableGeometryPool() const {
    const auto cesiumDataUsd = UsdUtil::getOrCreateCesiumData();
    bool disableGeometryPool;
    cesiumDataUsd.GetDebugDisableGeometryPoolAttr().Get(&disableGeometryPool);
    return disableGeometryPool;
}

bool Context::getDebugDisableMaterialPool() const {
    const auto cesiumDataUsd = UsdUtil::getOrCreateCesiumData();
    bool disableMaterialPool;
    cesiumDataUsd.GetDebugDisableMaterialPoolAttr().Get(&disableMaterialPool);
    return disableMaterialPool;
}

bool Context::getDebugDisableTexturePool() const {
    const auto cesiumDataUsd = UsdUtil::getOrCreateCesiumData();
    bool disableTexturePool;
    cesiumDataUsd.GetDebugDisableTexturePoolAttr().Get(&disableTexturePool);
    return disableTexturePool;
}

uint64_t Context::getDebugGeometryPoolInitialCapacity() const {
    const auto cesiumDataUsd = UsdUtil::getOrCreateCesiumData();
    uint64_t geometryPoolInitialCapacity;
    cesiumDataUsd.GetDebugGeometryPoolInitialCapacityAttr().Get(&geometryPoolInitialCapacity);
    return geometryPoolInitialCapacity;
}

uint64_t Context::getDebugMaterialPoolInitialCapacity() const {
    const auto cesiumDataUsd = UsdUtil::getOrCreateCesiumData();
    uint64_t materialPoolInitialCapacity;
    cesiumDataUsd.GetDebugMaterialPoolInitialCapacityAttr().Get(&materialPoolInitialCapacity);
    return materialPoolInitialCapacity;
}

uint64_t Context::getDebugTexturePoolInitialCapacity() const {
    const auto cesiumDataUsd = UsdUtil::getOrCreateCesiumData();
    uint64_t texturePoolInitialCapacity;
    cesiumDataUsd.GetDebugTexturePoolInitialCapacityAttr().Get(&texturePoolInitialCapacity);
    return texturePoolInitialCapacity;
}

bool Context::getDebugRandomColors() const {
    const auto cesiumDataUsd = UsdUtil::getOrCreateCesiumData();
    bool debugRandomColors;
    cesiumDataUsd.GetDebugRandomColorsAttr().Get(&debugRandomColors);
    return debugRandomColors;
}

bool Context::creditsAvailable() const {
    const auto& credits = _creditSystem->getCreditsToShowThisFrame();

    return credits.size() > 0;
}

std::vector<std::pair<std::string, bool>> Context::getCredits() const {
    const auto& credits = _creditSystem->getCreditsToShowThisFrame();

    std::vector<std::pair<std::string, bool>> result;
    result.reserve(credits.size());

    for (const auto& item : credits) {
        auto showOnScreen = _creditSystem->shouldBeShownOnScreen(item);
        result.emplace_back(_creditSystem->getHtml(item), showOnScreen);
    }

    return result;
}

void Context::creditsStartNextFrame() {
    _creditSystem->startNextFrame();
}

RenderStatistics Context::getRenderStatistics() const {
    RenderStatistics renderStatistics;

    auto fabricStatistics = FabricUtil::getStatistics();
    renderStatistics.materialsCapacity = fabricStatistics.materialsCapacity;
    renderStatistics.materialsLoaded = fabricStatistics.materialsLoaded;
    renderStatistics.geometriesCapacity = fabricStatistics.geometriesCapacity;
    renderStatistics.geometriesLoaded = fabricStatistics.geometriesLoaded;
    renderStatistics.geometriesRendered = fabricStatistics.geometriesRendered;
    renderStatistics.trianglesLoaded = fabricStatistics.trianglesLoaded;
    renderStatistics.trianglesRendered = fabricStatistics.trianglesRendered;

    const auto& tilesets = AssetRegistry::getInstance().getAllTilesets();
    for (const auto& tileset : tilesets) {
        auto tilesetStatistics = tileset->getStatistics();
        renderStatistics.tilesetCachedBytes += tilesetStatistics.tilesetCachedBytes;
        renderStatistics.tilesVisited += tilesetStatistics.tilesVisited;
        renderStatistics.culledTilesVisited += tilesetStatistics.culledTilesVisited;
        renderStatistics.tilesRendered += tilesetStatistics.tilesRendered;
        renderStatistics.tilesCulled += tilesetStatistics.tilesCulled;
        renderStatistics.maxDepthVisited += tilesetStatistics.maxDepthVisited;
        renderStatistics.tilesLoadingWorker += tilesetStatistics.tilesLoadingWorker;
        renderStatistics.tilesLoadingMain += tilesetStatistics.tilesLoadingMain;
        renderStatistics.tilesLoaded += tilesetStatistics.tilesLoaded;
    }

    return renderStatistics;
}

void Context::addGlobeAnchorToPrim(const pxr::SdfPath& path) {
    if (UsdUtil::isCesiumData(path) || UsdUtil::isCesiumGeoreference(path) || UsdUtil::isCesiumImagery(path) ||
        UsdUtil::isCesiumSession(path) || UsdUtil::isCesiumTileset(path)) {
        _logger->warn("Cannot attach Globe Anchor to Cesium Tilesets, Imagery, Georeference, Session, or Data prims.");
        return;
    }

    auto prim = UsdUtil::getUsdStage()->GetPrimAtPath(path);
    auto globeAnchor = UsdUtil::defineGlobeAnchor(path);

    // Until we support multiple georeference points, we should just use the default georeference object.
    auto georeferenceOrigin = UsdUtil::getOrCreateCesiumGeoreference();
    globeAnchor.GetGeoreferenceBindingRel().AddTarget(georeferenceOrigin.GetPath());

    const auto& cartographicOrigin = GeospatialUtil::convertGeoreferenceToCartographic(georeferenceOrigin);

    GeospatialUtil::updateAnchorByUsdTransform(cartographicOrigin, globeAnchor);
}

} // namespace cesium::omniverse
