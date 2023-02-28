#include "cesium/omniverse/Context.h"

#include "cesium/omniverse/AssetRegistry.h"
#include "cesium/omniverse/Broadcast.h"
#include "cesium/omniverse/CesiumIonSession.h"
#include "cesium/omniverse/HttpAssetAccessor.h"
#include "cesium/omniverse/LoggerSink.h"
#include "cesium/omniverse/OmniIonRasterOverlay.h"
#include "cesium/omniverse/OmniTileset.h"
#include "cesium/omniverse/TaskProcessor.h"
#include "cesium/omniverse/UsdUtil.h"

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <Cesium3DTilesSelection/CreditSystem.h>
#include <Cesium3DTilesSelection/Tileset.h>
#include <Cesium3DTilesSelection/ViewState.h>
#include <Cesium3DTilesSelection/registerAllTileContentTypes.h>
#include <CesiumUsdSchemas/data.h>
#include <CesiumUsdSchemas/rasterOverlay.h>
#include <CesiumUsdSchemas/tilesetAPI.h>
#include <CesiumUsdSchemas/tokens.h>
#include <glm/gtc/matrix_access.hpp>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdUtils/stageCache.h>

namespace cesium::omniverse {

namespace {

Cesium3DTilesSelection::ViewState computeViewState(
    const CesiumGeospatial::Cartographic& origin,
    const glm::dmat4& viewMatrix,
    const glm::dmat4& projMatrix,
    double width,
    double height) {
    const auto usdToEcef = UsdUtil::computeUsdToEcefTransform(origin);
    const auto inverseView = glm::inverse(viewMatrix);
    const auto omniCameraUp = glm::dvec3(viewMatrix[1]);
    const auto omniCameraFwd = glm::dvec3(-viewMatrix[2]);
    const auto omniCameraPosition = glm::dvec3(glm::row(inverseView, 3));
    const auto cameraUp = glm::normalize(glm::dvec3(usdToEcef * glm::dvec4(omniCameraUp, 0.0)));
    const auto cameraFwd = glm::normalize(glm::dvec3(usdToEcef * glm::dvec4(omniCameraFwd, 0.0)));
    const auto cameraPosition = glm::dvec3(usdToEcef * glm::dvec4(omniCameraPosition, 1.0));

    const auto aspect = width / height;
    const auto verticalFov = 2.0 * glm::atan(1.0 / projMatrix[1][1]);
    const auto horizontalFov = 2.0 * glm::atan(glm::tan(verticalFov * 0.5) * aspect);

    return Cesium3DTilesSelection::ViewState::create(
        cameraPosition, cameraFwd, cameraUp, glm::dvec2(width, height), horizontalFov, verticalFov);
}

const pxr::SdfPath CesiumDataPath{"/Cesium"};

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
    _memCesiumPath = _cesiumExtensionLocation / "bin" / "mem.cesium";
    _certificatePath = _cesiumExtensionLocation / "certs" / "cacert.pem";

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
}

void Context::destroy() {
    AssetRegistry::getInstance().clear();
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

void Context::addCesiumDataIfNotExists(const CesiumIonClient::Token& token) {
    if (!UsdUtil::primExists(CesiumDataPath)) {
        UsdUtil::defineCesiumData(CesiumDataPath);
    }

    if (!token.token.empty()) {
        auto cesiumDataUsd = UsdUtil::getCesiumData(CesiumDataPath);
        cesiumDataUsd.GetDefaultProjectIonAccessTokenAttr().Set<std::string>(token.token);
        cesiumDataUsd.GetDefaultProjectIonAccessTokenIdAttr().Set<std::string>(token.id);
    }
}

int64_t Context::addTilesetUrl(const std::string& url) {
    // Name actually needs to be something that we pass into this eventually.
    const auto tilesetId = _tilesetId++;
    const auto tilesetName = fmt::format("tileset_{}", tilesetId);
    const auto tilesetPath = UsdUtil::getPathUnique(UsdUtil::getRootPath(), tilesetName);
    const auto tilesetUsd = UsdUtil::defineCesiumTileset(tilesetPath);

    tilesetUsd.GetUrlAttr().Set<std::string>(url);

    tilesetUsd.GetMaximumScreenSpaceErrorAttr().Set<float>(16.0f);
    tilesetUsd.GetPreloadAncestorsAttr().Set<bool>(true);
    tilesetUsd.GetPreloadSiblingsAttr().Set<bool>(true);
    tilesetUsd.GetForbidHolesAttr().Set<bool>(false);
    tilesetUsd.GetMaximumSimultaneousTileLoadsAttr().Set<uint32_t>(20);
    tilesetUsd.GetMaximumCachedBytesAttr().Set<uint64_t>(536870912);
    tilesetUsd.GetLoadingDescendantLimitAttr().Set<uint32_t>(20);
    tilesetUsd.GetEnableFrustumCullingAttr().Set<bool>(true);
    tilesetUsd.GetEnableFogCullingAttr().Set<bool>(true);
    tilesetUsd.GetEnforceCulledScreenSpaceErrorAttr().Set<bool>(true);
    tilesetUsd.GetCulledScreenSpaceErrorAttr().Set<float>(64.0f);
    tilesetUsd.GetSuspendUpdateAttr().Set<bool>(false);

    AssetRegistry::getInstance().addTileset(tilesetId, tilesetPath);
    return tilesetId;
}

int64_t Context::addTilesetIon([[maybe_unused]] const std::string& name, int64_t ionId, const std::string& ionToken) {
    // Name actually needs to be something that we pass into this eventually.
    const auto tilesetId = _tilesetId++;
    const auto tilesetName = fmt::format("tileset_ion_{}", ionId);
    const auto tilesetPath = UsdUtil::getPathUnique(UsdUtil::getRootPath(), tilesetName);
    const auto tilesetUsd = UsdUtil::defineCesiumTileset(tilesetPath);

    tilesetUsd.GetIonAssetIdAttr().Set<int64_t>(ionId);
    tilesetUsd.GetIonAccessTokenAttr().Set<std::string>(ionToken);

    tilesetUsd.GetMaximumScreenSpaceErrorAttr().Set<float>(16.0f);
    tilesetUsd.GetPreloadAncestorsAttr().Set<bool>(true);
    tilesetUsd.GetPreloadSiblingsAttr().Set<bool>(true);
    tilesetUsd.GetForbidHolesAttr().Set<bool>(false);
    tilesetUsd.GetMaximumSimultaneousTileLoadsAttr().Set<uint32_t>(20);
    tilesetUsd.GetMaximumCachedBytesAttr().Set<uint64_t>(536870912);
    tilesetUsd.GetLoadingDescendantLimitAttr().Set<uint32_t>(20);
    tilesetUsd.GetEnableFrustumCullingAttr().Set<bool>(true);
    tilesetUsd.GetEnableFogCullingAttr().Set<bool>(true);
    tilesetUsd.GetEnforceCulledScreenSpaceErrorAttr().Set<bool>(true);
    tilesetUsd.GetCulledScreenSpaceErrorAttr().Set<float>(64.0f);
    tilesetUsd.GetSuspendUpdateAttr().Set<bool>(false);

    AssetRegistry::getInstance().addTileset(tilesetId, tilesetPath);
    return tilesetId;
}

void Context::addIonRasterOverlay(
    int64_t tilesetId,
    const std::string& name,
    int64_t ionId,
    const std::string& ionToken) {
    const auto tileset = AssetRegistry::getInstance().getTileset(tilesetId);

    if (!tileset.has_value()) {
        return;
    }

    const auto stage = UsdUtil::getUsdStage();
    const auto safeName = UsdUtil::getSafeName(name);
    auto path = UsdUtil::getPathUnique(tileset.value()->getPath(), safeName);
    auto rasterOverlayUsd = UsdUtil::defineCesiumRasterOverlay(path);

    rasterOverlayUsd.GetIonAssetIdAttr().Set<int64_t>(ionId);
    rasterOverlayUsd.GetIonAccessTokenAttr().Set<std::string>(ionToken);

    tileset.value()->addIonRasterOverlay(path);

    AssetRegistry::getInstance().addRasterOverlay(ionId, path, tilesetId);
}

void Context::removeTileset(int64_t tilesetId) {
    auto& assetRegistry = AssetRegistry::getInstance();
    const auto tileset = assetRegistry.getTileset(tilesetId);

    if (!tileset.has_value()) {
        return;
    }

    const auto stage = UsdUtil::getUsdStage();
    bool removed = stage->RemovePrim(tileset.value()->getPath());

    if (removed) {
        assetRegistry.removeAsset(tilesetId);
        assetRegistry.removeAssetByParent(tilesetId);
    }
}

void Context::reloadTileset(int64_t tilesetId) {
    const auto tileset = AssetRegistry::getInstance().getTileset(tilesetId);

    if (tileset.has_value()) {
        tileset.value()->reload();
    }
}

void Context::onUpdateFrame(const glm::dmat4& viewMatrix, const glm::dmat4& projMatrix, double width, double height) {
    processUsdNotifications();

    _viewStates.clear();
    _viewStates.emplace_back(computeViewState(_georeferenceOrigin, viewMatrix, projMatrix, width, height));

    auto tilesets = AssetRegistry::getInstance().getAllTilesets();
    for (const auto& tileset : tilesets) {
        tileset->onUpdateFrame(_viewStates);
    }
}

void Context::processUsdNotifications() {
    const auto changedProperties = _usdNotificationHandler.popChangedProperties();

    std::set<std::shared_ptr<OmniTileset>> tilesetsToReload;

    for (const auto& changedProperty : changedProperties) {
        const auto& [path, name, type] = changedProperty;

        if (type == ChangedPrimType::CESIUM_DATA) {
            if (name == pxr::CesiumTokens->cesiumDefaultProjectIonAccessToken) {
                // Any tilesets that use the default token are reloaded when it changes
                const auto tilesets = AssetRegistry::getInstance().getAllTilesets();
                for (const auto& tileset : tilesets) {
                    const auto tilesetToken = tileset->getIonAccessToken();
                    const auto defaultToken = Context::instance().getDefaultToken();
                    if (!tilesetToken.has_value() || tilesetToken.value().token == defaultToken.value().token) {
                        tilesetsToReload.emplace(tileset);
                    }
                }
            } else if (
                name == pxr::CesiumTokens->cesiumGeoreferenceOriginLongitude ||
                name == pxr::CesiumTokens->cesiumGeoreferenceOriginLatitude ||
                name == pxr::CesiumTokens->cesiumGeoreferenceOriginHeight) {
                const auto cesiumData = UsdUtil::getCesiumData(path);
                double longitude;
                double latitude;
                double height;
                cesiumData.GetGeoreferenceOriginLongitudeAttr().Get<double>(&longitude);
                cesiumData.GetGeoreferenceOriginLatitudeAttr().Get<double>(&latitude);
                cesiumData.GetGeoreferenceOriginHeightAttr().Get<double>(&height);

                setGeoreferenceOrigin(
                    CesiumGeospatial::Cartographic{glm::radians(longitude), glm::radians(latitude), height});
            }
        } else if (type == ChangedPrimType::CESIUM_TILESET) {
            // Reload the tileset. No need to update the asset registry because tileset assets do not store the asset id.
            const auto tileset = AssetRegistry::getInstance().getTileset(path.GetString());
            if (tileset.has_value()) {
                // clang-format off
                if (name == pxr::CesiumTokens->cesiumIonAssetId ||
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
                    name == pxr::CesiumTokens->cesiumCulledScreenSpaceError) {
                    tilesetsToReload.emplace(tileset.value());
                }
                // clang-format on
            }
        } else if (type == ChangedPrimType::CESIUM_RASTER_OVERLAY) {
            const auto tileset = AssetRegistry::getInstance().getTilesetFromRasterOverlay(path.GetString());
            if (tileset.has_value()) {
                if (name == pxr::CesiumTokens->cesiumIonAssetId) {
                    // Update the asset registry because the asset id changed
                    OmniIonRasterOverlay ionRasterOverlay(path);
                    const auto assetId = ionRasterOverlay.getIonAssetId();
                    AssetRegistry::getInstance().setRasterOverlayAssetId(path, assetId);

                    // Reload the tileset that this raster overlay is attached to
                    tilesetsToReload.emplace(tileset.value());
                } else if (name == pxr::CesiumTokens->cesiumIonAccessToken) {
                    // Reload the tileset that this raster overlay is attached to
                    tilesetsToReload.emplace(tileset.value());
                }
            }
        }
    }

    for (const auto& tileset : tilesetsToReload) {
        tileset->reload();
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

carb::flatcache::StageInProgress Context::getFabricStageInProgress() const {
    assert(_fabricStageInProgress.has_value());
    return _fabricStageInProgress.value();
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
        _fabricStageInProgress.reset();
        _stageId = 0;

        // Now it's safe to clear anything else that references the stage
        AssetRegistry::getInstance().clear();
    }

    if (newStage > 0) {
        // Set the USD stage
        _stage = pxr::UsdUtilsStageCache::Get().Find(pxr::UsdStageCache::Id::FromLongInt(stageId));

        // Set the Fabric stage
        const auto iStageInProgress = carb::getCachedInterface<carb::flatcache::IStageInProgress>();
        const auto stageInProgressId =
            iStageInProgress->get(carb::flatcache::UsdStageId{static_cast<uint64_t>(stageId)});
        _fabricStageInProgress = carb::flatcache::StageInProgress(stageInProgressId);
    }

    _stageId = stageId;
}

int64_t Context::getContextId() const {
    return _contextId;
}

const CesiumGeospatial::Cartographic& Context::getGeoreferenceOrigin() const {
    return _georeferenceOrigin;
}

void Context::setGeoreferenceOrigin(const CesiumGeospatial::Cartographic& origin) {
    _georeferenceOrigin = origin;
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
    if (!UsdUtil::primExists(CesiumDataPath)) {
        return std::nullopt;
    }

    const auto cesiumDataUsd = UsdUtil::getCesiumData(CesiumDataPath);
    std::string projectDefaultToken;
    cesiumDataUsd.GetDefaultProjectIonAccessTokenAttr().Get(&projectDefaultToken);
    std::string projectDefaultTokenId;
    cesiumDataUsd.GetDefaultProjectIonAccessTokenIdAttr().Get(&projectDefaultTokenId);

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
                addCesiumDataIfNotExists(response.value.value());

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
        addCesiumDataIfNotExists(token);

        _lastSetTokenResult =
            SetDefaultTokenResult{SetDefaultTokenResultCode::OK, SetDefaultTokenResultMessages::OK_MESSAGE};
    }

    Broadcast::setDefaultTokenComplete();
}
void Context::specifyToken(const std::string& token) {
    _session->findToken(token).thenInMainThread(
        [this, token](CesiumIonClient::Response<CesiumIonClient::Token>&& response) {
            if (response.value) {
                addCesiumDataIfNotExists(response.value.value());
            } else {
                CesiumIonClient::Token t;
                t.token = token;
                addCesiumDataIfNotExists(t);
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
void Context::updateTroubleshootingDetails(int64_t tilesetId, uint64_t tokenEventId, uint64_t assetEventId) {
    const auto tileset = AssetRegistry::getInstance().getTileset(tilesetId);

    if (!tileset.has_value()) {
        return;
    }

    TokenTroubleshooter troubleshooter;

    _assetTroubleshootingDetails = AssetTroubleshootingDetails();
    troubleshooter.updateAssetTroubleshootingDetails(tilesetId, assetEventId, _assetTroubleshootingDetails.value());

    _defaultTokenTroubleshootingDetails = TokenTroubleshootingDetails();

    if (isDefaultTokenSet()) {
        auto token = getDefaultToken().value().token;
        troubleshooter.updateTokenTroubleshootingDetails(
            tilesetId, token, tokenEventId, _defaultTokenTroubleshootingDetails.value());
    }

    // TODO: Implement grabbing data for the tileset token.
}
void Context::updateTroubleshootingDetails(
    int64_t tilesetId,
    int64_t rasterOverlayId,
    uint64_t tokenEventId,
    uint64_t assetEventId) {
    const auto tileset = AssetRegistry::getInstance().getTileset(tilesetId);

    if (!tileset.has_value()) {
        return;
    }

    TokenTroubleshooter troubleshooter;

    _assetTroubleshootingDetails = AssetTroubleshootingDetails();
    troubleshooter.updateAssetTroubleshootingDetails(
        rasterOverlayId, assetEventId, _assetTroubleshootingDetails.value());

    _defaultTokenTroubleshootingDetails = TokenTroubleshootingDetails();

    if (isDefaultTokenSet()) {
        auto token = getDefaultToken().value().token;
        troubleshooter.updateTokenTroubleshootingDetails(
            rasterOverlayId, token, tokenEventId, _defaultTokenTroubleshootingDetails.value());
    }

    // TODO: Implement grabbing data for the raster overlay token.
}

std::filesystem::path Context::getCesiumExtensionLocation() const {
    return _cesiumExtensionLocation;
}

std::filesystem::path Context::getMemCesiumPath() const {
    return _memCesiumPath;
}

std::filesystem::path Context::getCertificatePath() const {
    return _certificatePath;
}

bool Context::getDebugDisableMaterials() const {
    return _debugDisableMaterials;
}

} // namespace cesium::omniverse
