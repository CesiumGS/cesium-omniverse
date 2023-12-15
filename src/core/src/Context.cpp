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
#include "cesium/omniverse/OmniGlobeAnchor.h"
#include "cesium/omniverse/OmniImagery.h"
#include "cesium/omniverse/OmniTileset.h"
#include "cesium/omniverse/SessionRegistry.h"
#include "cesium/omniverse/TaskProcessor.h"
#include "cesium/omniverse/Tokens.h"
#include "cesium/omniverse/UsdUtil.h"

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <Cesium3DTilesContent/registerAllTileContentTypes.h>
#include <Cesium3DTilesSelection/Tileset.h>
#include <CesiumUsdSchemas/tokens.h>
#include <CesiumUtility/CreditSystem.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/primRange.h>
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
    _asyncSystem = std::make_shared<CesiumAsync::AsyncSystem>(_taskProcessor);
    _httpAssetAccessor = std::make_shared<HttpAssetAccessor>(_certificatePath);
    _creditSystem = std::make_shared<CesiumUtility::CreditSystem>();

    _logger = std::make_shared<spdlog::logger>(
        std::string("cesium-omniverse"),
        spdlog::sinks_init_list{
            std::make_shared<LoggerSink>(omni::log::Level::eVerbose),
            std::make_shared<LoggerSink>(omni::log::Level::eInfo),
            std::make_shared<LoggerSink>(omni::log::Level::eWarn),
            std::make_shared<LoggerSink>(omni::log::Level::eError),
            std::make_shared<LoggerSink>(omni::log::Level::eFatal),
        });

    Cesium3DTilesContent::registerAllTileContentTypes();

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

std::shared_ptr<CesiumAsync::AsyncSystem> Context::getAsyncSystem() {
    return _asyncSystem;
}

std::shared_ptr<HttpAssetAccessor> Context::getHttpAssetAccessor() {
    return _httpAssetAccessor;
}

std::shared_ptr<CesiumUtility::CreditSystem> Context::getCreditSystem() {
    return _creditSystem;
}

std::shared_ptr<spdlog::logger> Context::getLogger() {
    return _logger;
}

void Context::setProjectDefaultToken(const CesiumIonClient::Token& token) {
    if (token.token.empty()) {
        return;
    }

    const auto currentIonServer = UsdUtil::getOrCreateIonServer(UsdUtil::getPathToCurrentIonServer());

    currentIonServer.GetProjectDefaultIonAccessTokenAttr().Set<std::string>(token.token);
    currentIonServer.GetProjectDefaultIonAccessTokenIdAttr().Set<std::string>(token.id);
}

void Context::reloadTileset(const pxr::SdfPath& tilesetPath) {
    const auto tileset = AssetRegistry::getInstance().getTileset(tilesetPath);

    if (!tileset) {
        return;
    }

    tileset->reload();
}

void Context::clearStage() {
    // The order is important. Clear tilesets first so that Fabric resources are released back into the pool. Then clear the pools.
    AssetRegistry::getInstance().clear();
    FabricResourceManager::getInstance().clear();
    GlobeAnchorRegistry::getInstance().clear();
    SessionRegistry::getInstance().clear();
}

void Context::reloadStage() {
    clearStage();

    _usdNotificationHandler.onStageLoaded();

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

    const auto defaultIonServerPath = pxr::SdfPath("/CesiumServers/IonOfficial");

    // For backwards compatibility. Add default ion server to tilesets without a server.
    const auto& tilesets = AssetRegistry::getInstance().getAllTilesets();
    for (const auto& tileset : tilesets) {
        const auto ionServerPath = tileset->getIonServerPath();
        if (ionServerPath.IsEmpty()) {
            const auto tilesetPrim = UsdUtil::getCesiumTileset(tileset->getPath());
            tilesetPrim.GetIonServerBindingRel().SetTargets({defaultIonServerPath});
        }
    }

    // For backwards compatibility. Add default ion server to imagery without a server.
    const auto& imageries = AssetRegistry::getInstance().getAllImageries();
    for (const auto& imagery : imageries) {
        const auto ionServerPath = imagery->getIonServerPath();
        if (ionServerPath.IsEmpty()) {
            const auto imageryPrim = UsdUtil::getCesiumImagery(imagery->getPath());
            imageryPrim.GetIonServerBindingRel().SetTargets({defaultIonServerPath});
        }
    }
}

void Context::onUpdateFrame(const std::vector<Viewport>& viewports) {
    _usdNotificationHandler.onUpdateFrame();

    const auto georeferenceOrigin = Context::instance().getGeoreferenceOrigin();
    const auto ecefToUsdTransform = UsdUtil::computeEcefToUsdLocalTransform(georeferenceOrigin);

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

void Context::onUpdateUi() {
    // A lot of UI code will end up calling the session prior to us actually having a stage. The user won't see this
    // but some major segfaults will occur without this check.
    if (!UsdUtil::hasStage()) {
        return;
    }

    const auto sessions = SessionRegistry::getInstance().getAllSessions();

    for (const auto& session : sessions) {
        session->tick();
    }
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
    const auto session = SessionRegistry::getInstance().getSession(UsdUtil::getPathToCurrentIonServer());

    if (session == nullptr) {
        return;
    }

    session->connect();
}

std::optional<std::shared_ptr<CesiumIonSession>> Context::getSession() {
    // A lot of UI code will end up calling the session prior to us actually having a stage. The user won't see this
    // but some major segfaults will occur without this check.
    if (!UsdUtil::hasStage()) {
        return std::nullopt;
    }

    const auto session = SessionRegistry::getInstance().getSession(UsdUtil::getPathToCurrentIonServer());

    if (session == nullptr) {
        return std::nullopt;
    }

    return std::optional<std::shared_ptr<CesiumIonSession>>{session};
}

std::vector<std::shared_ptr<CesiumIonSession>> Context::getAllSessions() {
    // A lot of UI code will end up calling the session prior to us actually having a stage. The user won't see this
    // but some major segfaults will occur without this check.
    if (!UsdUtil::hasStage()) {
        return {};
    }

    return SessionRegistry::getInstance().getAllSessions();
}

std::optional<CesiumIonClient::Token> Context::getDefaultToken() const {
    const auto currentIonServer = UsdUtil::getOrCreateIonServer(UsdUtil::getPathToCurrentIonServer());

    std::string projectDefaultToken;
    std::string projectDefaultTokenId;

    currentIonServer.GetProjectDefaultIonAccessTokenAttr().Get(&projectDefaultToken);
    currentIonServer.GetProjectDefaultIonAccessTokenIdAttr().Get(&projectDefaultTokenId);

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
    const auto session = SessionRegistry::getInstance().getSession(UsdUtil::getPathToCurrentIonServer());
    auto connection = session->getConnection();

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
    const auto session = SessionRegistry::getInstance().getSession(UsdUtil::getPathToCurrentIonServer());
    auto connection = session->getConnection();

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
    const auto session = SessionRegistry::getInstance().getSession(UsdUtil::getPathToCurrentIonServer());
    session->findToken(token).thenInMainThread(
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
    const auto tileset = AssetRegistry::getInstance().getTileset(tilesetPath);

    if (!tileset) {
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

    auto tilesetIonAccessToken = tileset->getIonAccessToken();
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
    const auto tileset = registry.getTileset(tilesetPath);
    if (!tileset) {
        return;
    }

    const auto imagery = registry.getImageryByIonAssetId(imageryIonAssetId);
    if (!imagery) {
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

    auto imageryIonAccessToken = imagery->getIonAccessToken();
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
}

void Context::addGlobeAnchorToPrim(const pxr::SdfPath& path, double latitude, double longitude, double height) {
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

    pxr::GfVec3d coordinates{latitude, longitude, height};
    globeAnchor.GetGeographicCoordinateAttr().Set(coordinates);
}

} // namespace cesium::omniverse
