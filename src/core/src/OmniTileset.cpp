#include "cesium/omniverse/OmniTileset.h"

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
#include <glm/glm.hpp>

namespace cesium::omniverse {
static std::shared_ptr<TaskProcessor> taskProcessor;
static std::shared_ptr<HttpAssetAccessor> httpAssetAccessor;
static std::shared_ptr<Cesium3DTilesSelection::CreditSystem> creditSystem;
static uint64_t i = 0;

static uint64_t getID() {
    return i++;
}

OmniTileset::OmniTileset(const pxr::UsdStageRefPtr& stage, const std::string& url) {
    pxr::SdfPath tilesetPath =
        stage->GetPseudoRoot().GetPath().AppendChild(pxr::TfToken(fmt::format("tileset_{}", getID())));
    renderResourcesPreparer = std::make_shared<RenderResourcesPreparer>(stage, tilesetPath);
    CesiumAsync::AsyncSystem asyncSystem{taskProcessor};
    Cesium3DTilesSelection::TilesetExternals externals{
        httpAssetAccessor, renderResourcesPreparer, asyncSystem, creditSystem};

    initOriginShiftHandler();

    tileset = std::make_unique<Cesium3DTilesSelection::Tileset>(externals, url);
}

OmniTileset::OmniTileset(const pxr::UsdStageRefPtr& stage, int64_t ionID, const std::string& ionToken) {
    pxr::SdfPath tilesetPath =
        stage->GetPseudoRoot().GetPath().AppendChild(pxr::TfToken(fmt::format("tileset_ion_{}", ionID)));
    renderResourcesPreparer = std::make_shared<RenderResourcesPreparer>(stage, tilesetPath);
    CesiumAsync::AsyncSystem asyncSystem{taskProcessor};
    Cesium3DTilesSelection::TilesetExternals externals{
        httpAssetAccessor, renderResourcesPreparer, asyncSystem, creditSystem};

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
    Cesium3DTilesSelection::registerAllTileContentTypes();
}

void OmniTileset::shutdown() {
    taskProcessor.reset();
    httpAssetAccessor.reset();
    creditSystem.reset();
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
