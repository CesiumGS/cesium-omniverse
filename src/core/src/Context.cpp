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

void Context::setProjectDefaultToken(const CesiumIonClient::Token& token) {
    if (token.token.empty()) {
        return;
    }

    const auto cesiumDataUsd = UsdUtil::getOrCreateCesiumData();

    cesiumDataUsd.GetDefaultProjectIonAccessTokenAttr().Set<std::string>(token.token);
    cesiumDataUsd.GetDefaultProjectIonAccessTokenIdAttr().Set<std::string>(token.id);
}

pxr::SdfPath Context::addTilesetUrl(const std::string& name, const std::string& url) {
    const auto tilesetId = _tilesetId++;
    const auto tilesetName = UsdUtil::getSafeName(name);
    const auto tilesetPath = UsdUtil::getPathUnique(UsdUtil::getRootPath(), tilesetName);
    const auto tilesetUsd = UsdUtil::defineCesiumTileset(tilesetPath);

    tilesetUsd.GetUrlAttr().Set<std::string>(url);

    AssetRegistry::getInstance().addTileset(tilesetPath, tilesetId);
    return tilesetPath;
}

pxr::SdfPath Context::addTilesetIon(const std::string& name, int64_t ionAssetId, const std::string& ionAccessToken) {
    const auto tilesetId = _tilesetId++;
    const auto tilesetName = UsdUtil::getSafeName(name);
    const auto tilesetPath = UsdUtil::getPathUnique(UsdUtil::getRootPath(), tilesetName);
    const auto tilesetUsd = UsdUtil::defineCesiumTileset(tilesetPath);

    tilesetUsd.GetIonAssetIdAttr().Set<int64_t>(ionAssetId);
    tilesetUsd.GetIonAccessTokenAttr().Set<std::string>(ionAccessToken);

    AssetRegistry::getInstance().addTileset(tilesetPath, tilesetId);
    return tilesetPath;
}

pxr::SdfPath Context::addIonRasterOverlay(
    const pxr::SdfPath& tilesetPath,
    const std::string& name,
    int64_t ionAssetId,
    const std::string& ionAccessToken) {
    const auto tileset = AssetRegistry::getInstance().getTilesetByPath(tilesetPath);

    if (!tileset.has_value()) {
        return pxr::SdfPath();
    }

    const auto stage = UsdUtil::getUsdStage();
    const auto safeName = UsdUtil::getSafeName(name);
    auto path = UsdUtil::getPathUnique(tileset.value()->getPath(), safeName);
    auto rasterOverlayUsd = UsdUtil::defineCesiumRasterOverlay(path);

    rasterOverlayUsd.GetIonAssetIdAttr().Set<int64_t>(ionAssetId);
    rasterOverlayUsd.GetIonAccessTokenAttr().Set<std::string>(ionAccessToken);

    tileset.value()->addIonRasterOverlay(path);

    AssetRegistry::getInstance().addRasterOverlay(path);
    return path;
}

void Context::removeTileset(const pxr::SdfPath& tilesetPath) {
    auto& assetRegistry = AssetRegistry::getInstance();
    const auto tileset = assetRegistry.getTilesetByPath(tilesetPath);

    if (!tileset.has_value()) {
        return;
    }

    const auto stage = UsdUtil::getUsdStage();
    bool removed = !UsdUtil::primExists(tileset.value()->getPath());

    if (!removed) {
        removed = stage->RemovePrim(tileset.value()->getPath());
    }

    if (removed) {
        assetRegistry.removeTileset(tilesetPath);
    }
}

void Context::reloadTileset(const pxr::SdfPath& tilesetPath) {
    const auto tileset = AssetRegistry::getInstance().getTilesetByPath(tilesetPath);

    if (tileset.has_value()) {
        tileset.value()->reload();
    }
}

void Context::onUpdateFrame(const glm::dmat4& viewMatrix, const glm::dmat4& projMatrix, double width, double height) {
    processUsdNotifications();

    const auto georeferenceOrigin = getGeoreferenceOrigin();

    _viewStates.clear();
    _viewStates.emplace_back(computeViewState(georeferenceOrigin, viewMatrix, projMatrix, width, height));

    const auto& tilesets = AssetRegistry::getInstance().getAllTilesets();
    for (const auto& tileset : tilesets) {
        tileset->onUpdateFrame(_viewStates);
    }
}

void Context::processPropertyChanged(const ChangedPrim& changedProperty) {
    const auto& [path, name, primType, changeType] = changedProperty;

    std::set<std::shared_ptr<OmniTileset>> tilesetsToReload;

    if (primType == ChangedPrimType::CESIUM_DATA) {
        if (name == pxr::CesiumTokens->cesiumDefaultProjectIonAccessToken) {
            // Any tilesets that use the default token are reloaded when it changes
            const auto& tilesets = AssetRegistry::getInstance().getAllTilesets();
            for (const auto& tileset : tilesets) {
                const auto tilesetToken = tileset->getIonAccessToken();
                const auto defaultToken = Context::instance().getDefaultToken();
                if (!tilesetToken.has_value() || tilesetToken.value().token == defaultToken.value().token) {
                    tilesetsToReload.emplace(tileset);
                }
            }
        }
    } else if (primType == ChangedPrimType::CESIUM_TILESET) {
        // Reload the tileset. No need to update the asset registry because tileset assets do not store the asset id.
        const auto tileset = AssetRegistry::getInstance().getTilesetByPath(path);
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
    } else if (primType == ChangedPrimType::CESIUM_RASTER_OVERLAY) {
        const auto tilesetPath = path.GetParentPath();
        const auto tileset = AssetRegistry::getInstance().getTilesetByPath(tilesetPath);
        if (tileset.has_value()) {
            if (name == pxr::CesiumTokens->cesiumIonAssetId || name == pxr::CesiumTokens->cesiumIonAccessToken) {
                // Reload the tileset that this raster overlay is attached to
                tilesetsToReload.emplace(tileset.value());
            }
        }
    }

    for (const auto& tileset : tilesetsToReload) {
        tileset->reload();
    }
}

void Context::processPrimRemoved(const ChangedPrim& changedProperty) {
    if (changedProperty.primType == ChangedPrimType::CESIUM_TILESET) {
        removeTileset(changedProperty.path);
    } else if (changedProperty.primType == ChangedPrimType::CESIUM_RASTER_OVERLAY) {
        const auto tilesetPath = changedProperty.path.GetParentPath();
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

        // Add the CesiumData prim so that we can set the georeference origin and other top-level properties
        // without waiting for an ion session to start
        // TODO: does this clear the stage's previous georeference values?
        UsdUtil::getOrCreateCesiumData();
    }

    _stageId = stageId;
}

int64_t Context::getContextId() const {
    return _contextId;
}

const CesiumGeospatial::Cartographic Context::getGeoreferenceOrigin() const {
    const auto cesiumData = UsdUtil::getOrCreateCesiumData();

    double longitude;
    double latitude;
    double height;
    cesiumData.GetGeoreferenceOriginLongitudeAttr().Get<double>(&longitude);
    cesiumData.GetGeoreferenceOriginLatitudeAttr().Get<double>(&latitude);
    cesiumData.GetGeoreferenceOriginHeightAttr().Get<double>(&height);

    return CesiumGeospatial::Cartographic(glm::radians(longitude), glm::radians(latitude), height);
}

void Context::setGeoreferenceOrigin(const CesiumGeospatial::Cartographic& origin) {
    const auto cesiumData = UsdUtil::getOrCreateCesiumData();

    cesiumData.GetGeoreferenceOriginLongitudeAttr().Set<double>(glm::degrees(origin.longitude));
    cesiumData.GetGeoreferenceOriginLatitudeAttr().Set<double>(glm::degrees(origin.latitude));
    cesiumData.GetGeoreferenceOriginHeightAttr().Set<double>(origin.height);
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

    cesiumDataUsd.GetDefaultProjectIonAccessTokenAttr().Get(&projectDefaultToken);
    cesiumDataUsd.GetDefaultProjectIonAccessTokenIdAttr().Get(&projectDefaultTokenId);

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

    if (isDefaultTokenSet()) {
        auto defaultToken = getDefaultToken().value().token;
        troubleshooter.updateTokenTroubleshootingDetails(
            tilesetIonAssetId, defaultToken, tokenEventId, _defaultTokenTroubleshootingDetails.value());
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
    int64_t rasterOverlayIonAssetId,
    uint64_t tokenEventId,
    uint64_t assetEventId) {
    auto& registry = AssetRegistry::getInstance();
    const auto tileset = registry.getTilesetByPath(tilesetPath);
    if (!tileset.has_value()) {
        return;
    }

    const auto rasterOverlay = registry.getRasterOverlayByIonAssetId(rasterOverlayIonAssetId);
    if (!rasterOverlay.has_value()) {
        return;
    }

    TokenTroubleshooter troubleshooter;

    _assetTroubleshootingDetails = AssetTroubleshootingDetails();
    troubleshooter.updateAssetTroubleshootingDetails(
        rasterOverlayIonAssetId, assetEventId, _assetTroubleshootingDetails.value());

    _defaultTokenTroubleshootingDetails = TokenTroubleshootingDetails();

    if (isDefaultTokenSet()) {
        auto token = getDefaultToken().value().token;
        troubleshooter.updateTokenTroubleshootingDetails(
            rasterOverlayIonAssetId, token, tokenEventId, _defaultTokenTroubleshootingDetails.value());
    }

    _assetTokenTroubleshootingDetails = TokenTroubleshootingDetails();

    auto rasterOverlayToken = rasterOverlay.value()->getIonAccessToken();
    if (rasterOverlayToken.has_value()) {
        troubleshooter.updateTokenTroubleshootingDetails(
            rasterOverlayIonAssetId,
            rasterOverlayToken.value().token,
            tokenEventId,
            _assetTokenTroubleshootingDetails.value());
    }
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

bool Context::creditsAvailable() const {
    auto credits = _creditSystem->getCreditsToShowThisFrame();

    return credits.size() > 0;
}

std::vector<std::pair<std::string, bool>> Context::getCredits() const {
    auto credits = _creditSystem->getCreditsToShowThisFrame();

    std::vector<std::pair<std::string, bool>> result;
    result.reserve(credits.size());

    for (const auto& item : credits) {
        auto showOnScreen = _creditSystem->shouldBeShownOnScreen(item);
        result.emplace_back(_creditSystem->getHtml(item), showOnScreen);
    }

    return result;
}

} // namespace cesium::omniverse
