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
    bool hasImagery;
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
};

std::vector<IntermediaryMesh>
gatherMeshes(const OmniTileset& tileset, const glm::dmat4& tileTransform, const CesiumGltf::Model& model) {
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
        [tilesetId, tileId, &ecefToUsdTransform, &gltfToEcefTransform, smoothNormals, &meshes](
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
            });
        });

    return meshes;
}

std::vector<std::shared_ptr<FabricMesh>>
acquireFabricMeshes(const CesiumGltf::Model& model, const std::vector<IntermediaryMesh>& meshes, bool hasImagery) {
    CESIUM_TRACE("FabricPrepareRenderResources::acquireFabricMeshes");
    std::vector<std::shared_ptr<FabricMesh>> fabricMeshes;
    fabricMeshes.reserve(meshes.size());

    for (const auto& mesh : meshes) {
        const auto& primitive = model.meshes[mesh.meshId].primitives[mesh.primitiveId];
        fabricMeshes.emplace_back(
            FabricMeshManager::getInstance().acquireMesh(model, primitive, mesh.smoothNormals, hasImagery));
    }

    return fabricMeshes;
}

void setFabricMeshes(
    const CesiumGltf::Model& model,
    const std::vector<IntermediaryMesh>& meshes,
    const std::vector<std::shared_ptr<FabricMesh>>& fabricMeshes,
    bool hasImagery) {
    CESIUM_TRACE("FabricPrepareRenderResources::setFabricMeshes");
    for (size_t i = 0; i < meshes.size(); i++) {
            const mesh& intermediaryMesh = meshes[i];
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
            mesh.smoothNormals,
            hasImagery);
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

    const auto hasImagery = tileLoadResult.rasterOverlayDetails.has_value();

    return asyncSystem.createResolvedFuture(Cesium3DTilesSelection::TileLoadResultAndRenderResources{
        std::move(tileLoadResult),
        new TileLoadThreadResult{
            transform,
            hasImagery,
        },
    });
}

void* FabricPrepareRenderResources::prepareInMainThread(
    [[maybe_unused]] Cesium3DTilesSelection::Tile& tile,
    void* pLoadThreadResult) {
    if (!pLoadThreadResult) {
        return nullptr;
    }

    std::unique_ptr<TileLoadThreadResult> pTileLoadThreadResult{
        reinterpret_cast<TileLoadThreadResult*>(pLoadThreadResult)};

    const auto& tileTransform = pTileLoadThreadResult->tileTransform;
    const auto hasImagery = pTileLoadThreadResult->hasImagery;

    const auto& content = tile.getContent();
    auto pRenderContent = content.getRenderContent();
    if (!pRenderContent) {
        return nullptr;
    }

    const auto& model = pRenderContent->getModel();

    auto meshes = gatherMeshes(_tileset, tileTransform, model);
    auto fabricMeshes = acquireFabricMeshes(model, meshes, hasImagery);

    setFabricMeshes(model, meshes, fabricMeshes, hasImagery);

    return new TileRenderResources{
        tileTransform,
        std::move(fabricMeshes),
    };
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
    // Multiple overlays is not supported yet. The texture will get freed when the prim is freed.
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

    for (const auto& mesh : pTileRenderResources->fabricMeshes) {
        mesh->setImagery(&rasterTile.getImage(), translation, scale, overlayTextureCoordinateID);
    }
}

void FabricPrepareRenderResources::detachRasterInMainThread(
    [[maybe_unused]] const Cesium3DTilesSelection::Tile& tile,
    [[maybe_unused]] int32_t overlayTextureCoordinateID,
    [[maybe_unused]] const Cesium3DTilesSelection::RasterOverlayTile& rasterTile,
    [[maybe_unused]] void* pMainThreadRendererResources) noexcept {
    // Multiple overlays is not supported yet. The texture will get freed when the prim is freed.
}

} // namespace cesium::omniverse
