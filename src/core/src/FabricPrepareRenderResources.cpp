#include "cesium/omniverse/FabricPrepareRenderResources.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/FabricMesh.h"
#include "cesium/omniverse/FabricMeshManager.h"
#include "cesium/omniverse/OmniTileset.h"
#include "cesium/omniverse/UsdUtil.h"

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <Cesium3DTilesSelection/GltfUtilities.h>
#include <Cesium3DTilesSelection/Tile.h>
#include <Cesium3DTilesSelection/Tileset.h>
#include <CesiumAsync/AsyncSystem.h>

namespace cesium::omniverse {

namespace {
struct TileLoadThreadResult {
    glm::dmat4 tileTransform;
    std::vector<std::shared_ptr<FabricMesh>> fabricMeshes;
};

std::vector<std::shared_ptr<FabricMesh>> createFabricMeshes(
    const OmniTileset& tileset,
    const glm::dmat4& tileTransform,
    const CesiumGltf::Model& model,
    const CesiumGltf::ImageCesium* imagery,
    const glm::dvec2& imageryTexcoordTranslation,
    const glm::dvec2& imageryTexcoordScale,
    uint64_t imageryTexcoordSetIndex) {

    const auto tilesetId = tileset.getTilesetId();
    const auto tileId = Context::instance().getNextTileId();

    const auto smoothNormals = tileset.getSmoothNormals();

    const auto ecefToUsdTransform =
        UsdUtil::computeEcefToUsdTransformForPrim(Context::instance().getGeoreferenceOrigin(), tileset.getPath());

    auto gltfToEcefTransform = Cesium3DTilesSelection::GltfUtilities::applyRtcCenter(model, tileTransform);
    gltfToEcefTransform = Cesium3DTilesSelection::GltfUtilities::applyGltfUpAxisTransform(model, gltfToEcefTransform);

    std::vector<std::shared_ptr<FabricMesh>> fabricMeshes;

    model.forEachPrimitiveInScene(
        -1,
        [tilesetId,
         tileId,
         &ecefToUsdTransform,
         &gltfToEcefTransform,
         smoothNormals,
         imagery,
         &imageryTexcoordTranslation,
         &imageryTexcoordScale,
         imageryTexcoordSetIndex,
         &fabricMeshes](
            const CesiumGltf::Model& gltf,
            [[maybe_unused]] const CesiumGltf::Node& node,
            [[maybe_unused]] const CesiumGltf::Mesh& mesh,
            const CesiumGltf::MeshPrimitive& primitive,
            const glm::dmat4& transform) {
            const auto fabricMesh = FabricMeshManager::getInstance().acquireMesh(
                tilesetId,
                tileId,
                ecefToUsdTransform,
                gltfToEcefTransform,
                transform,
                gltf,
                primitive,
                smoothNormals,
                imagery,
                imageryTexcoordTranslation,
                imageryTexcoordScale,
                imageryTexcoordSetIndex);
            fabricMeshes.push_back(fabricMesh);
        });

    return fabricMeshes;
}

std::vector<std::shared_ptr<FabricMesh>>
createFabricMeshes(const OmniTileset& tileset, const glm::dmat4& tileTransform, const CesiumGltf::Model& model) {
    return createFabricMeshes(tileset, tileTransform, model, nullptr, glm::dvec2(), glm::dvec2(), 0);
}

} // namespace

FabricPrepareRenderResources::FabricPrepareRenderResources(const OmniTileset& tileset)
    : _tileset(tileset) {}

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

        // If there are no imagery layers attached to the tile add the tile right away
        if (!tileLoadResult.rasterOverlayDetails.has_value()) {

            const auto fabricMeshes = createFabricMeshes(_tileset, transform, *pModel);

            return asyncSystem.createResolvedFuture(Cesium3DTilesSelection::TileLoadResultAndRenderResources{
                std::move(tileLoadResult),
                new TileLoadThreadResult{
                    transform,
                    std::move(fabricMeshes),
                },
            });
        }

        // Otherwise add the tile + imagery later
        return asyncSystem.createResolvedFuture(Cesium3DTilesSelection::TileLoadResultAndRenderResources{
            std::move(tileLoadResult),
            new TileLoadThreadResult{
                transform,
                {},
            },
        });
    });
}

void* FabricPrepareRenderResources::prepareInMainThread(
    [[maybe_unused]] Cesium3DTilesSelection::Tile& tile,
    void* pLoadThreadResult) {
    if (pLoadThreadResult) {
        std::unique_ptr<TileLoadThreadResult> pTileLoadThreadResult{
            reinterpret_cast<TileLoadThreadResult*>(pLoadThreadResult)};
        return new TileRenderResources{
            pTileLoadThreadResult->tileTransform,
            std::move(pTileLoadThreadResult->fabricMeshes),
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

        for (const auto& mesh : pTileRenderResources->fabricMeshes) {
            FabricMeshManager::getInstance().releaseMesh(mesh);
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
    // Due to Kit 104.2 material limitations, a tile can only ever have one imagery attached.
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

    if (pTileRenderResources->fabricMeshes.size() > 0) {
        // Already created the tile with lower-res imagery.
        // Due to Kit 104.2 material limitations, we can't update the texture or assign a new material to the prim.
        // But we can delete the existing prim and create a new prim.
        for (const auto& mesh : pTileRenderResources->fabricMeshes) {
            FabricMeshManager::getInstance().releaseMesh(mesh);
        }
    }

    const auto fabricMeshes = createFabricMeshes(
        _tileset,
        pTileRenderResources->tileTransform,
        tile.getContent().getRenderContent()->getModel(),
        &rasterTile.getImage(),
        translation,
        scale,
        static_cast<uint64_t>(overlayTextureCoordinateID));

    pTileRenderResources->fabricMeshes = fabricMeshes;
}

void FabricPrepareRenderResources::detachRasterInMainThread(
    [[maybe_unused]] const Cesium3DTilesSelection::Tile& tile,
    [[maybe_unused]] int32_t overlayTextureCoordinateID,
    [[maybe_unused]] const Cesium3DTilesSelection::RasterOverlayTile& rasterTile,
    [[maybe_unused]] void* pMainThreadRendererResources) noexcept {
    // Nothing to do here.
    // Due to Kit 104.2 material limitations, a tile can only ever have one imagery attached.
    // If we remove the imagery from the tileset we need to reload the whole tileset.
}

} // namespace cesium::omniverse
