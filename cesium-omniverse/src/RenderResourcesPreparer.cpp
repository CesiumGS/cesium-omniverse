#include "RenderResourcesPreparer.h"

#include "GltfToUSD.h"

#include <Cesium3DTilesSelection/Tile.h>
#include <Cesium3DTilesSelection/TileID.h>
#include <pxr/usd/usd/attribute.h>
#include <pxr/usd/usdUtils/stitch.h>

#include <atomic>

// TODO: Determine if this is actually needed for anything.
// namespace pxr {
// TF_DEFINE_PRIVATE_TOKENS(_tileTokens, (visibility)(invisible)(visible));
// }

namespace Cesium {
static std::atomic<std::uint64_t> tileID;

void RenderResourcesPreparer::setTransform(const glm::dmat4& absToRelWorld) {
    pxr::GfMatrix4d currentTransform{
        absToRelWorld[0][0],
        absToRelWorld[0][1],
        absToRelWorld[0][2],
        absToRelWorld[0][3],
        absToRelWorld[1][0],
        absToRelWorld[1][1],
        absToRelWorld[1][2],
        absToRelWorld[1][3],
        absToRelWorld[2][0],
        absToRelWorld[2][1],
        absToRelWorld[2][2],
        absToRelWorld[2][3],
        absToRelWorld[3][0],
        absToRelWorld[3][1],
        absToRelWorld[3][2],
        absToRelWorld[3][3],
    };
    tilesetTransform.Set(currentTransform);
}

void RenderResourcesPreparer::setVisible(void* renderResources, bool enable) {
    if (renderResources) {
        TileRenderResources* tileRenderResources = reinterpret_cast<TileRenderResources*>(renderResources);
        if (enable != tileRenderResources->enable) {
            if (tileRenderResources->prim) {
                tileRenderResources->prim.SetActive(enable);
                tileRenderResources->enable = enable;
            }
        }
    }
}

CesiumAsync::Future<Cesium3DTilesSelection::TileLoadResultAndRenderResources>
RenderResourcesPreparer::prepareInLoadThread(
    const CesiumAsync::AsyncSystem& asyncSystem,
    Cesium3DTilesSelection::TileLoadResult&& tileLoadResult,
    const glm::dmat4& transform,
    [[maybe_unused]] const std::any& rendererOptions) {
    CesiumGltf::Model* pModel = std::get_if<CesiumGltf::Model>(&tileLoadResult.contentKind);
    if (!pModel)
        return asyncSystem.createResolvedFuture(
            Cesium3DTilesSelection::TileLoadResultAndRenderResources{std::move(tileLoadResult), nullptr});

    pxr::SdfLayerRefPtr anonLayer = pxr::SdfLayer::CreateAnonymous(".usda");
    pxr::UsdStageRefPtr anonStage = pxr::UsdStage::Open(anonLayer);
    auto prim = GltfToUSD::convertToUSD(
        anonStage,
        tilesetPath.AppendChild(pxr::TfToken(fmt::format("tile_{}", ++tileID))),
        *pModel,
        transform * CesiumGeometry::AxisTransforms::Y_UP_TO_Z_UP);
    prim.SetActive(false);

    return asyncSystem.createResolvedFuture(Cesium3DTilesSelection::TileLoadResultAndRenderResources{
        std::move(tileLoadResult), new TileWorkerRenderResources{std::move(anonLayer), prim.GetPath(), false}});
}

void* RenderResourcesPreparer::prepareInMainThread(
    [[maybe_unused]] Cesium3DTilesSelection::Tile& tile,
    void* pLoadThreadResult) {
    if (pLoadThreadResult) {
        std::unique_ptr<TileWorkerRenderResources> workerRenderResources{
            reinterpret_cast<TileWorkerRenderResources*>(pLoadThreadResult)};
        pxr::UsdUtilsStitchLayers(stage->GetRootLayer(), workerRenderResources->layer);
        return new TileRenderResources{stage->GetPrimAtPath(workerRenderResources->primPath), false};
    }

    return nullptr;
}

void RenderResourcesPreparer::free(
    [[maybe_unused]] Cesium3DTilesSelection::Tile& tile,
    void* pLoadThreadResult,
    void* pMainThreadResult) noexcept {
    if (pLoadThreadResult) {
        delete reinterpret_cast<TileWorkerRenderResources*>(pLoadThreadResult);
    }

    if (pMainThreadResult) {
        TileRenderResources* tileRenderResources = reinterpret_cast<TileRenderResources*>(pMainThreadResult);
        stage->RemovePrim(tileRenderResources->prim.GetPath());
        delete tileRenderResources;
    }
}

void* RenderResourcesPreparer::prepareRasterInLoadThread(
    [[maybe_unused]] CesiumGltf::ImageCesium& image,
    [[maybe_unused]] const std::any& rendererOptions) {
    return nullptr;
}

void* RenderResourcesPreparer::prepareRasterInMainThread(
    [[maybe_unused]] Cesium3DTilesSelection::RasterOverlayTile& rasterTile,
    [[maybe_unused]] void* pLoadThreadResult) {
    return nullptr;
}

void RenderResourcesPreparer::freeRaster(
    [[maybe_unused]] const Cesium3DTilesSelection::RasterOverlayTile& rasterTile,
    [[maybe_unused]] void* pLoadThreadResult,
    [[maybe_unused]] void* pMainThreadResult) noexcept {}

void RenderResourcesPreparer::attachRasterInMainThread(
    [[maybe_unused]] const Cesium3DTilesSelection::Tile& tile,
    [[maybe_unused]] int32_t overlayTextureCoordinateID,
    [[maybe_unused]] const Cesium3DTilesSelection::RasterOverlayTile& rasterTile,
    [[maybe_unused]] void* pMainThreadRendererResources,
    [[maybe_unused]] const glm::dvec2& translation,
    [[maybe_unused]] const glm::dvec2& scale) {}

void RenderResourcesPreparer::detachRasterInMainThread(
    [[maybe_unused]] const Cesium3DTilesSelection::Tile& tile,
    [[maybe_unused]] int32_t overlayTextureCoordinateID,
    [[maybe_unused]] const Cesium3DTilesSelection::RasterOverlayTile& rasterTile,
    [[maybe_unused]] void* pMainThreadRendererResources) noexcept {}
} // namespace Cesium
