#include "cesium/omniverse/OmniTileset.h"

#include "cesium/omniverse/Broadcast.h"
#include "cesium/omniverse/GltfToUSD.h"
#include "cesium/omniverse/HttpAssetAccessor.h"
#include "cesium/omniverse/LoggerSink.h"
#include "cesium/omniverse/TaskProcessor.h"

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <Cesium3DTilesSelection/IonRasterOverlay.h>
#include <Cesium3DTilesSelection/registerAllTileContentTypes.h>
#include <CesiumGeometry/AxisTransforms.h>
#include <CesiumGltf/Material.h>
#include <CesiumUsdSchemas/data.h>
#include <glm/glm.hpp>

namespace cesium::omniverse {
static std::shared_ptr<TaskProcessor> taskProcessor;
static std::shared_ptr<HttpAssetAccessor> httpAssetAccessor;
static std::shared_ptr<Cesium3DTilesSelection::CreditSystem> creditSystem;
static std::shared_ptr<CesiumIonSession> session;
static pxr::UsdStageRefPtr usdStage;
static SetDefaultTokenResult lastSetTokenResult;
static uint64_t i = 0;

static uint64_t getID() {
    return i++;
}

OmniTileset::OmniTileset(const std::string& url) {
    pxr::SdfPath tilesetPath =
        usdStage->GetPseudoRoot().GetPath().AppendChild(pxr::TfToken(fmt::format("tileset_{}", getID())));
    renderResourcesPreparer = std::make_shared<RenderResourcesPreparer>(usdStage, tilesetPath);
    CesiumAsync::AsyncSystem asyncSystem{taskProcessor};
    Cesium3DTilesSelection::TilesetExternals externals{
        httpAssetAccessor, renderResourcesPreparer, asyncSystem, creditSystem};

    auto tilesetApi = OmniTileset::applyTilesetApiToPath(tilesetPath);
    tilesetApi.GetTilesetUrlAttr().Set<std::string>(url);

    initOriginShiftHandler();

    tileset = std::make_unique<Cesium3DTilesSelection::Tileset>(externals, url);
}

OmniTileset::OmniTileset(int64_t ionID, const std::string& ionToken) {
    pxr::SdfPath tilesetPath =
        usdStage->GetPseudoRoot().GetPath().AppendChild(pxr::TfToken(fmt::format("tileset_ion_{}", ionID)));
    renderResourcesPreparer = std::make_shared<RenderResourcesPreparer>(usdStage, tilesetPath);
    CesiumAsync::AsyncSystem asyncSystem{taskProcessor};
    Cesium3DTilesSelection::TilesetExternals externals{
        httpAssetAccessor, renderResourcesPreparer, asyncSystem, creditSystem};

    auto tilesetApi = OmniTileset::applyTilesetApiToPath(tilesetPath);
    tilesetApi.GetTilesetIdAttr().Set<int64_t>(ionID);

    initOriginShiftHandler();

    tileset = std::make_unique<Cesium3DTilesSelection::Tileset>(externals, ionID, ionToken);
}

void OmniTileset::updateFrame(
    const pxr::GfMatrix4d& viewMatrix,
    const pxr::GfMatrix4d& projMatrix,
    double width,
    double height) {
    viewStates.clear();
    if (tileset) {
        const glm::dmat4& relToAbs = Georeference::instance().relToAbsWorld;
        auto inverseView = viewMatrix.GetInverse();
        pxr::GfVec4d omniCameraUp = inverseView.GetRow(1);
        pxr::GfVec4d omniCameraFwd = -inverseView.GetRow(2);
        pxr::GfVec4d omniCameraPosition = inverseView.GetRow(3);
        glm::dvec3 cameraUp{relToAbs * glm::dvec4(omniCameraUp[0], omniCameraUp[1], omniCameraUp[2], 0.0)};
        glm::dvec3 cameraFwd{relToAbs * glm::dvec4(omniCameraFwd[0], omniCameraFwd[1], omniCameraFwd[2], 0.0)};
        glm::dvec3 cameraPosition{
            relToAbs * glm::dvec4(omniCameraPosition[0], omniCameraPosition[1], omniCameraPosition[2], 1.0)};

        cameraUp = glm::normalize(cameraUp);
        cameraFwd = glm::normalize(cameraFwd);

        double aspect = width / height;
        double verticalFov = 2.0 * glm::atan(1.0 / projMatrix[1][1]);
        double horizontalFov = 2.0 * glm::atan(glm::tan(verticalFov * 0.5) * aspect);

        auto viewState = Cesium3DTilesSelection::ViewState::create(
            cameraPosition, cameraFwd, cameraUp, glm::dvec2(width, height), horizontalFov, verticalFov);
        viewStates.emplace_back(viewState);
        tileset->getOptions().enableFrustumCulling = false;
        tileset->getOptions().forbidHoles = true;
        tileset->getOptions().maximumSimultaneousTileLoads = 10;
        tileset->getOptions().loadingDescendantLimit = 10;
        const auto& viewUpdate = tileset->updateView(viewStates);

        // retrieve tiles are visible in the current frame
        for (Cesium3DTilesSelection::Tile* tile : viewUpdate.tilesFadingOut) {
            if (tile->getState() == Cesium3DTilesSelection::TileLoadState::Done) {
                auto renderContent = tile->getContent().getRenderContent();
                if (renderContent) {
                    void* renderResources = renderContent->getRenderResources();
                    renderResourcesPreparer->setVisible(renderResources, false);
                }
            }
        }

        for (Cesium3DTilesSelection::Tile* tile : viewUpdate.tilesToRenderThisFrame) {
            if (tile->getState() == Cesium3DTilesSelection::TileLoadState::Done) {
                auto renderContent = tile->getContent().getRenderContent();
                if (renderContent) {
                    void* renderResources = renderContent->getRenderResources();
                    renderResourcesPreparer->setVisible(renderResources, true);
                }
            }
        }
    }
}

void OmniTileset::addIonRasterOverlay(const std::string& name, int64_t ionId, const std::string& ionToken) {
    Cesium3DTilesSelection::RasterOverlayOptions options;
    options.loadErrorCallback = [](const Cesium3DTilesSelection::RasterOverlayLoadFailureDetails& error) {
        spdlog::default_logger()->error("Raster overlay failed");
        spdlog::default_logger()->error(error.message);
    };

    rasterOverlay = new Cesium3DTilesSelection::IonRasterOverlay(name, ionId, ionToken, options);
    tileset->getOverlays().add(rasterOverlay);
}

void OmniTileset::init(const std::filesystem::path& cesiumExtensionLocation) {
    GltfToUSD::CesiumMemLocation = cesiumExtensionLocation / "bin";
#ifdef CESIUM_OMNI_UNIX
    HttpAssetAccessor::CertificatePath = cesiumExtensionLocation / "certs";
#endif

    auto logger = spdlog::default_logger();
    logger->sinks().clear();
    logger->sinks().push_back(std::make_shared<LoggerSink>());

    taskProcessor = std::make_shared<TaskProcessor>();
    httpAssetAccessor = std::make_shared<HttpAssetAccessor>();
    creditSystem = std::make_shared<Cesium3DTilesSelection::CreditSystem>();
    CesiumAsync::AsyncSystem asyncSystem{taskProcessor};
    session = std::make_shared<CesiumIonSession>(asyncSystem, httpAssetAccessor);
    session->resume();
    Cesium3DTilesSelection::registerAllTileContentTypes();
}

pxr::CesiumTilesetAPI OmniTileset::applyTilesetApiToPath(const pxr::SdfPath& path) {
    auto prim = usdStage->GetPrimAtPath(path);
    auto tilesetApi = pxr::CesiumTilesetAPI::Apply(prim);

    tilesetApi.CreateTilesetUrlAttr(pxr::VtValue(""));
    tilesetApi.CreateTilesetIdAttr(pxr::VtValue(""));
    tilesetApi.CreateNameAttr(pxr::VtValue(""));
    tilesetApi.CreateIonTokenAttr(pxr::VtValue(""));

    return tilesetApi;
}

std::optional<CesiumIonClient::Token> OmniTileset::getDefaultToken() {
    pxr::UsdPrim cesiumDataPrim = usdStage->GetPrimAtPath(pxr::SdfPath("/Cesium"));

    if (!cesiumDataPrim.IsValid()) {
        return {};
    }

    pxr::CesiumData cesiumData(cesiumDataPrim);
    std::string projectDefaultToken;
    cesiumData.GetDefaultProjectTokenAttr().Get(&projectDefaultToken);
    std::string projectDefaultTokenId;
    cesiumData.GetDefaultProjectTokenIdAttr().Get(&projectDefaultTokenId);

    return CesiumIonClient::Token{projectDefaultTokenId, "", projectDefaultToken};
}

[[maybe_unused]] pxr::UsdStageRefPtr& OmniTileset::getStage() {
    return usdStage;
}

void OmniTileset::setStage(const pxr::UsdStageRefPtr& stage) {
    usdStage = stage;
}

void OmniTileset::shutdown() {
    taskProcessor.reset();
    httpAssetAccessor.reset();
    creditSystem.reset();
    session.reset();
    usdStage.Reset();
}

/**
 * Adds the Cesium prim to the stage if it doesn't already exist and sets the default project token to the set value.
 * If you want to set the default project token to something else, use this function.
 *
 * @param token The default project token in string form.
 */
void OmniTileset::addCesiumDataIfNotExists(const CesiumIonClient::Token& token) {
    pxr::SdfPath sdfPath = pxr::SdfPath("/Cesium");
    pxr::UsdPrim cesiumDataPrim = usdStage->GetPrimAtPath(sdfPath);
    if (!cesiumDataPrim.IsValid()) {
        cesiumDataPrim = usdStage->DefinePrim(sdfPath);
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

void OmniTileset::connectToIon() {
    if (session == nullptr) {
        return;
    }

    session->connect();
}

SetDefaultTokenResult OmniTileset::getSetDefaultTokenResult() {
    return lastSetTokenResult;
}

void OmniTileset::createToken(const std::string& name) {
    auto connection = session->getConnection();

    if (!connection.has_value()) {
        lastSetTokenResult = SetDefaultTokenResult{
            SetDefaultTokenResultCode::NOT_CONNECTED_TO_ION,
            SetDefaultTokenResultMessages::NOT_CONNECTED_TO_ION_MESSAGE};
        return;
    }

    connection->createToken(name, {"assets:read"}, std::vector<int64_t>{1}, std::nullopt)
        .thenInMainThread([](CesiumIonClient::Response<CesiumIonClient::Token>&& response) {
            if (response.value) {
                addCesiumDataIfNotExists(response.value.value());

                lastSetTokenResult =
                    SetDefaultTokenResult{SetDefaultTokenResultCode::OK, SetDefaultTokenResultMessages::OK_MESSAGE};
            } else {
                lastSetTokenResult = SetDefaultTokenResult{
                    SetDefaultTokenResultCode::CREATE_FAILED,
                    fmt::format(
                        SetDefaultTokenResultMessages::CREATE_FAILED_MESSAGE_BASE,
                        response.errorMessage,
                        response.errorCode)};
            }

            Broadcast::setDefaultTokenComplete();
        });
}

void OmniTileset::selectToken(const CesiumIonClient::Token& token) {
    auto connection = session->getConnection();

    if (!connection.has_value()) {
        lastSetTokenResult = SetDefaultTokenResult{
            SetDefaultTokenResultCode::NOT_CONNECTED_TO_ION,
            SetDefaultTokenResultMessages::NOT_CONNECTED_TO_ION_MESSAGE};
    } else {
        addCesiumDataIfNotExists(token);

        lastSetTokenResult =
            SetDefaultTokenResult{SetDefaultTokenResultCode::OK, SetDefaultTokenResultMessages::OK_MESSAGE};
    }

    Broadcast::setDefaultTokenComplete();
}

void OmniTileset::specifyToken(const std::string& token) {
    session->findToken(token).thenInMainThread([token](CesiumIonClient::Response<CesiumIonClient::Token>&& response) {
        if (response.value) {
            addCesiumDataIfNotExists(response.value.value());
        } else {
            CesiumIonClient::Token t;
            t.token = token;
            addCesiumDataIfNotExists(t);
        }
        // We assume the user knows what they're doing if they specify a token not on their account.
        lastSetTokenResult =
            SetDefaultTokenResult{SetDefaultTokenResultCode::OK, SetDefaultTokenResultMessages::OK_MESSAGE};

        Broadcast::setDefaultTokenComplete();
    });
}

void OmniTileset::onUiUpdate() {
    if (session == nullptr) {
        return;
    }

    session->tick();
}

std::optional<std::shared_ptr<CesiumIonSession>> OmniTileset::getSession() {
    if (session == nullptr) {
        return std::nullopt;
    }

    return std::optional<std::shared_ptr<CesiumIonSession>>{session};
}

void OmniTileset::initOriginShiftHandler() {
    originShiftHandler.set_callback(
        [this]([[maybe_unused]] const glm::dmat4& relToAbsWorld, const glm::dmat4& absToRelWorld) {
            this->renderResourcesPreparer->setTransform(absToRelWorld);
        });

    originShiftHandler.connect(Georeference::instance().originChangeEvent);
    this->renderResourcesPreparer->setTransform(Georeference::instance().absToRelWorld);
}
} // namespace cesium::omniverse
