#include "cesium/omniverse/FabricPrepareRenderResources.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/FabricStageUtil.h"
#include "cesium/omniverse/OmniTileset.h"
#include "cesium/omniverse/UsdUtil.h"

#include <Cesium3DTilesSelection/Tile.h>
#include <Cesium3DTilesSelection/Tileset.h>
#include <CesiumAsync/AsyncSystem.h>

namespace cesium::omniverse {

namespace {
struct TileLoadThreadResult {
    int64_t tileId;
    glm::dmat4 tileTransform;
    std::vector<pxr::SdfPath> geomPaths;
    std::vector<pxr::SdfPath> allPrimPaths;
    std::vector<std::string> textureAssetNames;
};
} // namespace

FabricPrepareRenderResources::FabricPrepareRenderResources(const OmniTileset& tileset)
    : _tileset(tileset) {}

FabricPrepareRenderResources::~FabricPrepareRenderResources() {
    if (UsdUtil::hasStage()) {
        FabricStageUtil::removeTileset(_tileset.getId());
    }
}

CesiumAsync::Future<Cesium3DTilesSelection::TileLoadResultAndRenderResources>
FabricPrepareRenderResources::prepareInLoadThread(
    const CesiumAsync::AsyncSystem& asyncSystem,
    Cesium3DTilesSelection::TileLoadResult&& tileLoadResult,
    const glm::dmat4& transform,
    [[maybe_unused]] const std::any& rendererOptions) {
    const auto pModel = std::get_if<CesiumGltf::Model>(&tileLoadResult.contentKind);
    if (!pModel) {
        return asyncSystem.createResolvedFuture(
            Cesium3DTilesSelection::TileLoadResultAndRenderResources{std::move(tileLoadResult), nullptr});
    }

    return asyncSystem.runInMainThread([this, asyncSystem, transform, tileLoadResult = std::move(tileLoadResult)]() {
        const auto pModel = std::get_if<CesiumGltf::Model>(&tileLoadResult.contentKind);

        const auto tileId = _tileset.getNextTileId();

        // If there are no raster overlays add the tile right away
        if (!tileLoadResult.rasterOverlayDetails.has_value()) {
            const auto ecefToUsdTransform = UsdUtil::computeEcefToUsdTransformForPrim(
                Context::instance().getGeoreferenceOrigin(), _tileset.getPath());

            const auto addTileResults =
                FabricStageUtil::addTile(_tileset.getId(), tileId, ecefToUsdTransform, transform, *pModel);

            return asyncSystem.createResolvedFuture(Cesium3DTilesSelection::TileLoadResultAndRenderResources{
                std::move(tileLoadResult),
                new TileLoadThreadResult{
                    tileId,
                    transform,
                    std::move(addTileResults.geomPaths),
                    std::move(addTileResults.allPrimPaths),
                    std::move(addTileResults.textureAssetNames),
                }});
        }

        // Otherwise add the tile + raster overlay later
        return asyncSystem.createResolvedFuture(Cesium3DTilesSelection::TileLoadResultAndRenderResources{
            std::move(tileLoadResult),
            new TileLoadThreadResult{
                tileId,
                transform,
                {},
                {},
                {},
            }});
    });
}

void* FabricPrepareRenderResources::prepareInMainThread(
    [[maybe_unused]] Cesium3DTilesSelection::Tile& tile,
    void* pLoadThreadResult) {
    if (pLoadThreadResult) {
        std::unique_ptr<TileLoadThreadResult> pTileLoadThreadResult{
            reinterpret_cast<TileLoadThreadResult*>(pLoadThreadResult)};
        return new TileRenderResources{
            pTileLoadThreadResult->tileId,
            pTileLoadThreadResult->tileTransform,
            std::move(pTileLoadThreadResult->geomPaths),
            std::move(pTileLoadThreadResult->allPrimPaths),
            std::move(pTileLoadThreadResult->textureAssetNames),
        };
    }

    return nullptr;
}

void FabricPrepareRenderResources::free(
    [[maybe_unused]] Cesium3DTilesSelection::Tile& tile,
    void* pLoadThreadResult,
    void* pMainThreadResult) noexcept {
    if (pLoadThreadResult) {
        const auto pTileLoadThreadResult = reinterpret_cast<TileLoadThreadResult*>(pLoadThreadResult);
        delete pTileLoadThreadResult;
    }

    if (pMainThreadResult) {
        const auto pTileRenderResources = reinterpret_cast<TileRenderResources*>(pMainThreadResult);

        if (UsdUtil::hasStage()) {
            FabricStageUtil::removeTile(pTileRenderResources->allPrimPaths, pTileRenderResources->textureAssetNames);
        }

        delete pTileRenderResources;
    }
}

void* FabricPrepareRenderResources::prepareRasterInLoadThread(
    [[maybe_unused]] CesiumGltf::ImageCesium& image,
    [[maybe_unused]] const std::any& rendererOptions) {
    return nullptr;
}

void* FabricPrepareRenderResources::prepareRasterInMainThread(
    [[maybe_unused]] Cesium3DTilesSelection::RasterOverlayTile& rasterTile,
    [[maybe_unused]] void* pLoadThreadResult) {
    return nullptr;
}

void FabricPrepareRenderResources::freeRaster(
    [[maybe_unused]] const Cesium3DTilesSelection::RasterOverlayTile& rasterTile,
    [[maybe_unused]] void* pLoadThreadResult,
    [[maybe_unused]] void* pMainThreadResult) noexcept {
    // Nothing to do here.
    // Due to Kit 104.2 material limitations, a tile can only ever have one raster attached.
    // The texture will get freed when the prim is freed.
}

void FabricPrepareRenderResources::attachRasterInMainThread(
    const Cesium3DTilesSelection::Tile& tile,
    int32_t overlayTextureCoordinateID,
    const Cesium3DTilesSelection::RasterOverlayTile& rasterTile,
    [[maybe_unused]] void* pMainThreadRendererResources,
    const glm::dvec2& translation,
    const glm::dvec2& scale) {
    auto& content = tile.getContent();
    auto pRenderContent = content.getRenderContent();
    if (!pRenderContent) {
        return;
    }

    auto pTileRenderResources = reinterpret_cast<TileRenderResources*>(pRenderContent->getRenderResources());
    if (!pTileRenderResources) {
        return;
    }

    if (pTileRenderResources->geomPaths.size() > 0) {
        // Already created the tile with a lower-res raster.
        // Due to Kit 104.2 material limitations, we can't update the texture or assign a new material to the prim.
        // We are stuck with the low res raster.
        return;
    }

    const auto ecefToUsdTransform =
        UsdUtil::computeEcefToUsdTransformForPrim(Context::instance().getGeoreferenceOrigin(), _tileset.getPath());

    const auto addTileResults = FabricStageUtil::addTileWithRasterOverlay(
        _tileset.getId(),
        pTileRenderResources->tileId,
        ecefToUsdTransform,
        pTileRenderResources->tileTransform,
        tile.getContent().getRenderContent()->getModel(),
        rasterTile.getImage(),
        rasterTile.getOverlay().getName(),
        rasterTile.getRectangle(),
        translation,
        scale,
        static_cast<uint64_t>(overlayTextureCoordinateID));

    pTileRenderResources->geomPaths = addTileResults.geomPaths;
    pTileRenderResources->allPrimPaths = addTileResults.allPrimPaths;
    pTileRenderResources->textureAssetNames = addTileResults.textureAssetNames;
}

void FabricPrepareRenderResources::detachRasterInMainThread(
    [[maybe_unused]] const Cesium3DTilesSelection::Tile& tile,
    [[maybe_unused]] int32_t overlayTextureCoordinateID,
    [[maybe_unused]] const Cesium3DTilesSelection::RasterOverlayTile& rasterTile,
    [[maybe_unused]] void* pMainThreadRendererResources) noexcept {
    // Nothing to do here.
    // Due to Kit 104.2 material limitations, a tile can only ever have one raster attached.
    // If we remove the raster overlay from the tileset we need to reload the whole tileset.
}

} // namespace cesium::omniverse
