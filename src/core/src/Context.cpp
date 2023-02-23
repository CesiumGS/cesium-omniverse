#include "cesium/omniverse/Context.h"

#include "cesium/omniverse/Broadcast.h"
#include "cesium/omniverse/CesiumIonSession.h"
#include "cesium/omniverse/HttpAssetAccessor.h"
#include "cesium/omniverse/LoggerSink.h"
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
#include <glm/gtc/matrix_access.hpp>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usdGeom/xform.h>

namespace cesium::omniverse {

namespace {
OmniTileset* findTileset(const std::vector<std::unique_ptr<OmniTileset>>& tilesets, int64_t tilesetId) {
    auto iter = std::find_if(
        tilesets.begin(), tilesets.end(), [&tilesetId](const auto& tileset) { return tileset->getId() == tilesetId; });

    if (iter != tilesets.end()) {
        return iter->get();
    }

    return nullptr;
}

void removeTileset(std::vector<std::unique_ptr<OmniTileset>>& tilesets, int64_t tilesetId) {
    auto removedIter = std::remove_if(
        tilesets.begin(), tilesets.end(), [&tilesetId](const auto& tileset) { return tileset->getId() == tilesetId; });

    tilesets.erase(removedIter, tilesets.end());
}

pxr::CesiumTilesetAPI applyTilesetApiToPath(const pxr::SdfPath& path) {
    auto stage = UsdUtil::getUsdStage();
    auto prim = stage->GetPrimAtPath(path);
    auto tilesetApi = pxr::CesiumTilesetAPI::Apply(prim);

    tilesetApi.CreateTilesetUrlAttr();
    tilesetApi.CreateTilesetIdAttr();
    tilesetApi.CreateIonTokenAttr();

    return tilesetApi;
}

pxr::CesiumRasterOverlay applyRasterOverlayToPath(const pxr::SdfPath& path) {
    auto stage = UsdUtil::getUsdStage();
    auto prim = stage->GetPrimAtPath(path);
    pxr::CesiumRasterOverlay rasterOverlay(prim);

    rasterOverlay.CreateIonTokenAttr();
    rasterOverlay.CreateRasterOverlayIdAttr();

    return rasterOverlay;
}

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

    if (context) {
        context->destroy();
    }
    context = std::make_unique<Context>(contextId++, cesiumExtensionLocation);
}

void Context::onShutdown() {
    if (context) {
        context->destroy();
    }
    context.reset();
}

void Context::onStageChange(long stageId) {
    // This works for now because all it does is destroy the tilesets but we
    // need a more robust approach for stage changes
    context->destroy();
    context->setStageId(stageId);
}

Context& Context::instance() {
    return *context.get();
}

Context::Context(int64_t contextId, const std::filesystem::path& cesiumExtensionLocation)
    : _contextId(contextId) {

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
    _tilesets.clear();
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
    auto stage = UsdUtil::getUsdStage();
    pxr::UsdPrim cesiumDataPrim = stage->GetPrimAtPath(CesiumDataPath);
    if (!cesiumDataPrim.IsValid()) {
        cesiumDataPrim = stage->DefinePrim(CesiumDataPath);
    }

    pxr::CesiumData cesiumData(cesiumDataPrim);
    auto projectDefaultToken = cesiumData.GetDefaultProjectTokenAttr();
    auto projectDefaultTokenId = cesiumData.GetDefaultProjectTokenIdAttr();

    if (!projectDefaultToken.IsValid()) {
        projectDefaultToken = cesiumData.CreateDefaultProjectTokenAttr(pxr::VtValue(""));
        projectDefaultTokenId = cesiumData.CreateDefaultProjectTokenIdAttr(pxr::VtValue(""));
    }

    if (!token.token.empty()) {
        projectDefaultToken.Set(token.token.c_str());
        projectDefaultTokenId.Set(token.id.c_str());
    }
}

int64_t Context::addTilesetUrl(const std::string& url) {
    // Name actually needs to be something that we pass into this eventually.
    const auto tilesetId = _tilesetId++;
    const auto tilesetName = fmt::format("tileset_{}", tilesetId);
    const auto tilesetPath = UsdUtil::getPathUnique(UsdUtil::getRootPath(), tilesetName);
    const auto stage = UsdUtil::getUsdStage();
    pxr::UsdGeomXform::Define(stage, tilesetPath);

    auto tilesetApi = applyTilesetApiToPath(tilesetPath);
    tilesetApi.GetTilesetUrlAttr().Set<std::string>(url);

    _tilesets.emplace_back(std::make_unique<OmniTileset>(tilesetId, tilesetPath));
    return tilesetId;
}

int64_t Context::addTilesetIon([[maybe_unused]] const std::string& name, int64_t ionId, const std::string& ionToken) {
    // Name actually needs to be something that we pass into this eventually.
    const auto tilesetId = _tilesetId++;
    const auto tilesetName = fmt::format("tileset_ion_{}", ionId);
    const auto tilesetPath = UsdUtil::getPathUnique(UsdUtil::getRootPath(), tilesetName);
    const auto stage = UsdUtil::getUsdStage();
    pxr::UsdGeomXform::Define(stage, tilesetPath);

    auto tilesetApi = applyTilesetApiToPath(tilesetPath);
    tilesetApi.GetTilesetIdAttr().Set<int64_t>(ionId);
    tilesetApi.GetIonTokenAttr().Set<std::string>(ionToken);

    _tilesets.emplace_back(std::make_unique<OmniTileset>(tilesetId, tilesetPath));
    return tilesetId;
}

void Context::addIonRasterOverlay(
    int64_t tilesetId,
    const std::string& name,
    int64_t ionId,
    const std::string& ionToken) {
    const auto tileset = findTileset(_tilesets, tilesetId);

    if (!tileset) {
        return;
    }

    const auto stage = UsdUtil::getUsdStage();
    const auto safeName = UsdUtil::getSafeName(name);
    auto path = UsdUtil::getPathUnique(tileset->getPath(), safeName);
    auto prim = stage->DefinePrim(path);

    // In the event that there is an issue with the prim, it will be invalid. This prevents a segfault.
    if (!prim.IsValid()) {
        CESIUM_LOG_ERROR("Raster Overlay control prim definition failed.");
        return;
    }

    auto rasterOverlay = applyRasterOverlayToPath(path);
    rasterOverlay.GetRasterOverlayIdAttr().Set<int64_t>(ionId);
    rasterOverlay.GetIonTokenAttr().Set<std::string>(ionToken);

    tileset->addIonRasterOverlay(path);
}

std::vector<std::pair<int64_t, const char*>> Context::getAllTilesetIdsAndPaths() const {
    std::vector<std::pair<int64_t, const char*>> result;
    result.reserve(_tilesets.size());

    for (const auto& tileset : _tilesets) {
        result.emplace_back(tileset->getId(), tileset->getPath().GetText());
    }

    return result;
}

void Context::removeTileset(int64_t tilesetId) {
    const auto tileset = findTileset(_tilesets, tilesetId);

    if (!tileset) {
        return;
    }

    const auto stage = UsdUtil::getUsdStage();
    stage->RemovePrim(tileset->getPath());

    ::cesium::omniverse::removeTileset(_tilesets, tilesetId);
}

void Context::reloadTileset(int64_t tilesetId) {
    const auto tileset = findTileset(_tilesets, tilesetId);

    if (tileset) {
        tileset->reload();
    }
}

void Context::onUpdateFrame(const glm::dmat4& viewMatrix, const glm::dmat4& projMatrix, double width, double height) {
    _viewStates.clear();
    _viewStates.emplace_back(computeViewState(_georeferenceOrigin, viewMatrix, projMatrix, width, height));

    for (const auto& tileset : _tilesets) {
        tileset->onUpdateFrame(_viewStates);
    }
}

void Context::onUpdateUi() {
    if (_session == nullptr) {
        return;
    }

    _session->tick();
}

long Context::getStageId() const {
    return _stageId;
}

void Context::setStageId(long stageId) {
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
    auto stage = UsdUtil::getUsdStage();
    auto cesiumDataPrim = stage->GetPrimAtPath(CesiumDataPath);

    if (!cesiumDataPrim.IsValid()) {
        return std::nullopt;
    }

    const pxr::CesiumData cesiumData(cesiumDataPrim);
    std::string projectDefaultToken;
    cesiumData.GetDefaultProjectTokenAttr().Get(&projectDefaultToken);
    std::string projectDefaultTokenId;
    cesiumData.GetDefaultProjectTokenIdAttr().Get(&projectDefaultTokenId);

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
    const auto tileset = findTileset(_tilesets, tilesetId);

    if (!tileset) {
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
    const auto tileset = findTileset(_tilesets, tilesetId);

    if (!tileset) {
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
