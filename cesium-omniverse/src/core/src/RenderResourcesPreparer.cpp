#include "cesium/omniverse/RenderResourcesPreparer.h"

#include "cesium/omniverse/GltfToUSD.h"

#include <Cesium3DTilesSelection/Tile.h>
#include <Cesium3DTilesSelection/TileID.h>
#include <pxr/usd/usd/attribute.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdUtils/stitch.h>

#include <atomic>

// TODO: Determine if this is actually needed for anything.
// namespace pxr {
// TF_DEFINE_PRIVATE_TOKENS(_tileTokens, (visibility)(invisible)(visible));
// }

namespace Cesium {
static std::atomic<std::uint64_t> tileID;

RenderResourcesPreparer::RenderResourcesPreparer(const pxr::UsdStageRefPtr& stage_, const pxr::SdfPath& tilesetPath_)
    : stage{stage_}
    , tilesetPath{tilesetPath_} {
    auto xform = pxr::UsdGeomXform::Get(stage, tilesetPath_);
    if (xform) {
        auto prim = xform.GetPrim();
        stage->RemovePrim(prim.GetPath());
    }

    xform = pxr::UsdGeomXform::Define(stage, tilesetPath_);

    const glm::dmat4& z_to_y = CesiumGeometry::AxisTransforms::Z_UP_TO_Y_UP;
    pxr::GfMatrix4d currentTransform{
        z_to_y[0][0],
        z_to_y[0][1],
        z_to_y[0][2],
        z_to_y[0][3],
        z_to_y[1][0],
        z_to_y[1][1],
        z_to_y[1][2],
        z_to_y[1][3],
        z_to_y[2][0],
        z_to_y[2][1],
        z_to_y[2][2],
        z_to_y[2][3],
        z_to_y[3][0],
        z_to_y[3][1],
        z_to_y[3][2],
        z_to_y[3][3],
    };
    tilesetTransform = xform.AddTransformOp();
    tilesetTransform.Set(currentTransform);
}

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

    // It is not possible for multiple threads to simulatenously write to the same stage, but it is safe for different
    // threads to write simultaneously to different stages, which is why we write to an anonymous stage in the load
    // thread and merge with the main stage in the main thread. See
    // https://graphics.pixar.com/usd/release/api/_usd__page__multi_threading.html
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
    CesiumGltf::ImageCesium& image,
    [[maybe_unused]] const std::any& rendererOptions) {
    // We don't have access to the tile path so the best we can do is convert the image to a BMP and insert it into the
    // memory asset cache later
    return new RasterRenderResources{GltfToUSD::writeImageToBmp(image)};
}

void* RenderResourcesPreparer::prepareRasterInMainThread(
    [[maybe_unused]] Cesium3DTilesSelection::RasterOverlayTile& rasterTile,
    void* pLoadThreadResult) {
    if (pLoadThreadResult) {
        // We don't have access to the tile path here either so simply pass along the result of
        // prepareRasterInLoadThread
        return pLoadThreadResult;
    }

    return nullptr;
}

void RenderResourcesPreparer::freeRaster(
    [[maybe_unused]] const Cesium3DTilesSelection::RasterOverlayTile& rasterTile,
    void* pLoadThreadResult,
    void* pMainThreadResult) noexcept {
    if (pLoadThreadResult) {
        delete reinterpret_cast<RasterRenderResources*>(pLoadThreadResult);
    }

    if (pMainThreadResult) {
        delete reinterpret_cast<RasterRenderResources*>(pMainThreadResult);
    }
}

void RenderResourcesPreparer::attachRasterInMainThread(
    const Cesium3DTilesSelection::Tile& tile,
    [[maybe_unused]] int32_t overlayTextureCoordinateID,
    [[maybe_unused]] const Cesium3DTilesSelection::RasterOverlayTile& rasterTile,
    void* pMainThreadRendererResources,
    [[maybe_unused]] const glm::dvec2& translation,
    [[maybe_unused]] const glm::dvec2& scale) {
    auto& content = tile.getContent();
    auto pRenderContent = content.getRenderContent();
    if (!pRenderContent) {
        return;
    }

    auto pTileRenderResources = reinterpret_cast<TileRenderResources*>(pRenderContent->getRenderResources());
    if (!pTileRenderResources) {
        return;
    }

    auto pRasterRenderResources = reinterpret_cast<RasterRenderResources*>(pMainThreadRendererResources);
    if (!pRasterRenderResources) {
        return;
    }

    const auto modelPath = pTileRenderResources->prim.GetPath();
    GltfToUSD::insertRasterOverlayTexture(modelPath, std::move(pRasterRenderResources->image));
}

void RenderResourcesPreparer::detachRasterInMainThread(
    [[maybe_unused]] const Cesium3DTilesSelection::Tile& tile,
    [[maybe_unused]] int32_t overlayTextureCoordinateID,
    [[maybe_unused]] const Cesium3DTilesSelection::RasterOverlayTile& rasterTile,
    [[maybe_unused]] void* pMainThreadRendererResources) noexcept {}
} // namespace Cesium
