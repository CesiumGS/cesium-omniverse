#include "cesium/omniverse/FabricPrepareRenderResources.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/FabricMesh.h"
#include "cesium/omniverse/FabricMeshManager.h"
#include "cesium/omniverse/GeospatialUtil.h"
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

template <typename T> size_t getIndexFromRef(const std::vector<T>& vector, const T& item) {
    return static_cast<size_t>(&item - vector.data());
};

struct TileLoadThreadResult {
    glm::dmat4 tileTransform;
    std::vector<std::shared_ptr<FabricMesh>> fabricMeshes;
};

struct IntermediaryMesh {
    const int64_t tilesetId;
    const int64_t tileId;
    const glm::dmat4 ecefToUsdTransform;
    const glm::dmat4 gltfToEcefTransform;
    const glm::dmat4 nodeTransform;
    const uint64_t meshId;
    const uint64_t primitiveId;
    const bool smoothNormals;
    const CesiumGltf::ImageCesium* imagery;
    const glm::dvec2 imageryTexcoordTranslation;
    const glm::dvec2 imageryTexcoordScale;
    const uint64_t imageryTexcoordSetIndex;
};

std::vector<IntermediaryMesh> gatherMeshes(
    const OmniTileset& tileset,
    const glm::dmat4& tileTransform,
    const CesiumGltf::Model& model,
    const CesiumGltf::ImageCesium* imagery,
    const glm::dvec2& imageryTexcoordTranslation,
    const glm::dvec2& imageryTexcoordScale,
    uint64_t imageryTexcoordSetIndex) {

    CESIUM_TRACE("FabricPrepareRenderResources::gatherMeshes");
    const auto tilesetId = tileset.getTilesetId();
    const auto tileId = Context::instance().getNextTileId();

    const auto smoothNormals = tileset.getSmoothNormals();

    const auto georeferenceOrigin = GeospatialUtil::convertGeoreferenceToCartographic(tileset.getGeoreference());
    const auto ecefToUsdTransform = UsdUtil::computeEcefToUsdTransformForPrim(georeferenceOrigin, tileset.getPath());

    auto gltfToEcefTransform = Cesium3DTilesSelection::GltfUtilities::applyRtcCenter(model, tileTransform);
    gltfToEcefTransform = Cesium3DTilesSelection::GltfUtilities::applyGltfUpAxisTransform(model, gltfToEcefTransform);

    std::vector<IntermediaryMesh> meshes;

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
         &meshes](
            const CesiumGltf::Model& gltf,
            [[maybe_unused]] const CesiumGltf::Node& node,
            [[maybe_unused]] const CesiumGltf::Mesh& mesh,
            const CesiumGltf::MeshPrimitive& primitive,
            const glm::dmat4& transform) {
            const auto meshId = getIndexFromRef(gltf.meshes, mesh);
            const auto primitiveId = getIndexFromRef(mesh.primitives, primitive);
            meshes.emplace_back(IntermediaryMesh{
                tilesetId,
                tileId,
                ecefToUsdTransform,
                gltfToEcefTransform,
                transform,
                meshId,
                primitiveId,
                smoothNormals,
                imagery,
                imageryTexcoordTranslation,
                imageryTexcoordScale,
                imageryTexcoordSetIndex,
            });
        });

    return meshes;
}

std::vector<IntermediaryMesh>
gatherMeshes(const OmniTileset& tileset, const glm::dmat4& tileTransform, const CesiumGltf::Model& model) {
    return gatherMeshes(tileset, tileTransform, model, nullptr, glm::dvec2(), glm::dvec2(), 0);
}

std::vector<std::shared_ptr<FabricMesh>>
acquireFabricMeshes(const CesiumGltf::Model& model, const std::vector<IntermediaryMesh>& meshes) {
    CESIUM_TRACE("FabricPrepareRenderResources::acquireFabricMeshes");
    std::vector<std::shared_ptr<FabricMesh>> fabricMeshes;
    fabricMeshes.reserve(meshes.size());

    for (const auto& mesh : meshes) {
        const auto& primitive = model.meshes[mesh.meshId].primitives[mesh.primitiveId];
        fabricMeshes.emplace_back(FabricMeshManager::getInstance().acquireMesh(
            model, primitive, mesh.smoothNormals, mesh.imagery, mesh.imageryTexcoordSetIndex));
    }

    return fabricMeshes;
}

void setFabricMeshes(
    const CesiumGltf::Model& model,
    const std::vector<IntermediaryMesh>& meshes,
    const std::vector<std::shared_ptr<FabricMesh>>& fabricMeshes) {
    CESIUM_TRACE("FabricPrepareRenderResources::setFabricMeshes");
    for (size_t i = 0; i < meshes.size(); i++) {
            const IntermediaryMesh& intermediaryMesh = meshes[i];
        const auto& fabricMesh = fabricMeshes[i];
        const auto& primitive = model.meshes[intermediaryMesh.meshId].primitives[intermediaryMesh.primitiveId];
        fabricMesh->setTile(
            intermediaryMesh.tilesetId,
            intermediaryMesh.tileId,
            intermediaryMesh.ecefToUsdTransform,
            intermediaryMesh.gltfToEcefTransform,
            intermediaryMesh.nodeTransform,
            model,
            primitive,
            intermediaryMesh.smoothNormals,
            intermediaryMesh.imagery,
            intermediaryMesh.imageryTexcoordTranslation,
            intermediaryMesh.imageryTexcoordScale,
            intermediaryMesh.imageryTexcoordSetIndex);
    }
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

    // If there are no imagery layers attached to the tile add the tile right away
    if (!tileLoadResult.rasterOverlayDetails.has_value()) {
        auto meshes = gatherMeshes(_tileset, transform, *pModel);
        return asyncSystem.runInMainThread(
            [transform, meshes = std::move(meshes), tileLoadResult = std::move(tileLoadResult)]() {
                const auto& model = *std::get_if<CesiumGltf::Model>(&tileLoadResult.contentKind);
                auto fabricMeshes = acquireFabricMeshes(model, meshes);
                setFabricMeshes(model, meshes, fabricMeshes);
                return Cesium3DTilesSelection::TileLoadResultAndRenderResources{
                    std::move(tileLoadResult),
                    new TileLoadThreadResult{
                        transform,
                        std::move(fabricMeshes),
                    },
                };
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

        for (const auto& mesh : pTileLoadThreadResult->fabricMeshes) {
            FabricMeshManager::getInstance().releaseMesh(mesh);
        }

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

    const auto& model = tile.getContent().getRenderContent()->getModel();

    const auto meshes = gatherMeshes(
        _tileset,
        pTileRenderResources->tileTransform,
        model,
        &rasterTile.getImage(),
        translation,
        scale,
        static_cast<uint64_t>(overlayTextureCoordinateID));

    const auto fabricMeshes = acquireFabricMeshes(model, meshes);
    setFabricMeshes(model, meshes, fabricMeshes);

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
