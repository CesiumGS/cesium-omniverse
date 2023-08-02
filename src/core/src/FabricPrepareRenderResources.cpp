#include "cesium/omniverse/FabricPrepareRenderResources.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/FabricGeometry.h"
#include "cesium/omniverse/FabricMaterial.h"
#include "cesium/omniverse/FabricResourceManager.h"
#include "cesium/omniverse/FabricTexture.h"
#include "cesium/omniverse/GeospatialUtil.h"
#include "cesium/omniverse/GltfUtil.h"
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
#include <omni/fabric/FabricUSD.h>
#include <omni/ui/ImageProvider/DynamicTextureProvider.h>

namespace cesium::omniverse {

namespace {

template <typename T> size_t getIndexFromRef(const std::vector<T>& vector, const T& item) {
    return static_cast<size_t>(&item - vector.data());
};

struct ImageryLoadThreadResult {
    std::shared_ptr<FabricTexture> texture;
};

struct ImageryRenderResources {
    std::shared_ptr<FabricTexture> texture;
};

struct MeshInfo {
    const int64_t tilesetId;
    const glm::dmat4 ecefToUsdTransform;
    const glm::dmat4 gltfToEcefTransform;
    const glm::dmat4 nodeTransform;
    const uint64_t meshId;
    const uint64_t primitiveId;
    const bool smoothNormals;
};

struct TileLoadThreadResult {
    std::vector<MeshInfo> meshes;
    std::vector<FabricMesh> fabricMeshes;
    glm::dmat4 tileTransform;
    bool hasImagery;
};

std::vector<MeshInfo>
gatherMeshes(const OmniTileset& tileset, const glm::dmat4& tileTransform, const CesiumGltf::Model& model) {
    CESIUM_TRACE("FabricPrepareRenderResources::gatherMeshes");
    const auto tilesetId = tileset.getTilesetId();

    const auto smoothNormals = tileset.getSmoothNormals();

    const auto georeferenceOrigin = GeospatialUtil::convertGeoreferenceToCartographic(tileset.getGeoreference());
    const auto ecefToUsdTransform = UsdUtil::computeEcefToUsdTransformForPrim(georeferenceOrigin, tileset.getPath());

    auto gltfToEcefTransform = Cesium3DTilesSelection::GltfUtilities::applyRtcCenter(model, tileTransform);
    gltfToEcefTransform = Cesium3DTilesSelection::GltfUtilities::applyGltfUpAxisTransform(model, gltfToEcefTransform);

    std::vector<MeshInfo> meshes;

    model.forEachPrimitiveInScene(
        -1,
        [tilesetId, &ecefToUsdTransform, &gltfToEcefTransform, smoothNormals, &meshes](
            const CesiumGltf::Model& gltf,
            [[maybe_unused]] const CesiumGltf::Node& node,
            const CesiumGltf::Mesh& mesh,
            const CesiumGltf::MeshPrimitive& primitive,
            const glm::dmat4& transform) {
            const auto meshId = getIndexFromRef(gltf.meshes, mesh);
            const auto primitiveId = getIndexFromRef(mesh.primitives, primitive);
            meshes.emplace_back(MeshInfo{
                tilesetId,
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

std::vector<FabricMesh> acquireFabricMeshes(
    const CesiumGltf::Model& model,
    const std::vector<MeshInfo>& meshes,
    bool hasImagery,
    const OmniTileset& tileset) {
    CESIUM_TRACE("FabricPrepareRenderResources::acquireFabricMeshes");
    std::vector<FabricMesh> fabricMeshes;
    fabricMeshes.reserve(meshes.size());

    auto& fabricResourceManager = FabricResourceManager::getInstance();
    const auto stageId = UsdUtil::getUsdStageId();

    for (const auto& mesh : meshes) {
        auto& fabricMesh = fabricMeshes.emplace_back();

        const auto& primitive = model.meshes[mesh.meshId].primitives[mesh.primitiveId];
        const auto fabricGeometry =
            fabricResourceManager.acquireGeometry(model, primitive, mesh.smoothNormals, stageId);
        fabricMesh.geometry = fabricGeometry;

        const auto shouldAcquireMaterial = FabricResourceManager::getInstance().shouldAcquireMaterial(
            primitive, hasImagery, tileset.getMaterialPath());

        if (shouldAcquireMaterial) {
            const auto materialInfo = GltfUtil::getMaterialInfo(model, primitive);

            const auto fabricMaterial = fabricResourceManager.acquireMaterial(materialInfo, hasImagery, stageId);

            fabricMesh.material = fabricMaterial;
            fabricMesh.materialInfo = materialInfo;

            if (fabricMaterial->getMaterialDefinition().hasBaseColorTexture() &&
                materialInfo.baseColorTexture.has_value()) {
                const auto fabricTexture = fabricResourceManager.acquireTexture();
                fabricMesh.baseColorTexture = fabricTexture;
            }
        }
    }

    return fabricMeshes;
}

void setFabricTextures(
    const CesiumGltf::Model& model,
    const std::vector<MeshInfo>& meshes,
    std::vector<FabricMesh>& fabricMeshes) {
    CESIUM_TRACE("FabricPrepareRenderResources::setFabricTextures");
    for (size_t i = 0; i < meshes.size(); i++) {
        const auto& meshInfo = meshes[i];
        const auto& primitive = model.meshes[meshInfo.meshId].primitives[meshInfo.primitiveId];
        auto& mesh = fabricMeshes[i];
        auto& baseColorTexture = mesh.baseColorTexture;

        if (baseColorTexture != nullptr) {
            const auto baseColorTextureImage = GltfUtil::getBaseColorTextureImage(model, primitive);
            baseColorTexture->setImage(*baseColorTextureImage);
        }
    }
}

void setFabricMeshes(
    const CesiumGltf::Model& model,
    const std::vector<MeshInfo>& meshes,
    std::vector<FabricMesh>& fabricMeshes,
    bool hasImagery,
    const OmniTileset& tileset) {
    CESIUM_TRACE("FabricPrepareRenderResources::setFabricMeshes");
    for (size_t i = 0; i < meshes.size(); i++) {
        const auto& meshInfo = meshes[i];
        const auto& primitive = model.meshes[meshInfo.meshId].primitives[meshInfo.primitiveId];
        const auto& materialPath = tileset.getMaterialPath();

        auto& mesh = fabricMeshes[i];
        auto& geometry = mesh.geometry;
        auto& material = mesh.material;
        auto& baseColorTexture = mesh.baseColorTexture;
        auto& materialInfo = mesh.materialInfo;

        geometry->setGeometry(
            meshInfo.tilesetId,
            meshInfo.ecefToUsdTransform,
            meshInfo.gltfToEcefTransform,
            meshInfo.nodeTransform,
            model,
            primitive,
            meshInfo.smoothNormals,
            hasImagery);

        if (material != nullptr) {
            material->setMaterial(meshInfo.tilesetId, materialInfo);
            geometry->setMaterial(material->getPathFabric());

            if (baseColorTexture != nullptr && materialInfo.baseColorTexture.has_value()) {
                material->setBaseColorTexture(baseColorTexture, materialInfo.baseColorTexture.value());
            }
        } else if (!materialPath.IsEmpty()) {
            geometry->setMaterial(omni::fabric::Path(omni::fabric::asInt(materialPath)));
        }
    }
}

void freeFabricMeshes(const std::vector<FabricMesh>& fabricMeshes) {
    auto& fabricResourceManager = FabricResourceManager::getInstance();

    for (const auto& mesh : fabricMeshes) {
        auto& geometry = mesh.geometry;
        auto& material = mesh.material;
        auto& baseColorTexture = mesh.baseColorTexture;

        assert(geometry != nullptr);

        fabricResourceManager.releaseGeometry(geometry);

        if (material != nullptr) {
            fabricResourceManager.releaseMaterial(material);
        }

        if (baseColorTexture != nullptr) {
            fabricResourceManager.releaseTexture(baseColorTexture);
        }
    }
}

} // namespace

FabricPrepareRenderResources::FabricPrepareRenderResources(const OmniTileset& tileset)
    : _tileset(&tileset) {}

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

    if (!tilesetExists()) {
        return asyncSystem.createResolvedFuture(
            Cesium3DTilesSelection::TileLoadResultAndRenderResources{std::move(tileLoadResult), nullptr});
    }

    const auto hasImagery = tileLoadResult.rasterOverlayDetails.has_value();

    auto meshes = gatherMeshes(*_tileset, transform, *pModel);

    struct IntermediateLoadThreadResult {
        Cesium3DTilesSelection::TileLoadResult tileLoadResult;
        std::vector<MeshInfo> meshes;
        std::vector<FabricMesh> fabricMeshes;
    };

    return asyncSystem
        .runInMainThread(
            [this, hasImagery, meshes = std::move(meshes), tileLoadResult = std::move(tileLoadResult)]() mutable {
                if (!tilesetExists()) {
                    return IntermediateLoadThreadResult{
                        std::move(tileLoadResult),
                        {},
                        {},
                    };
                }

                const auto pModel = std::get_if<CesiumGltf::Model>(&tileLoadResult.contentKind);
                auto fabricMeshes = acquireFabricMeshes(*pModel, meshes, hasImagery, *_tileset);
                return IntermediateLoadThreadResult{
                    std::move(tileLoadResult),
                    std::move(meshes),
                    std::move(fabricMeshes),
                };
            })
        .thenInWorkerThread([this, transform, hasImagery](IntermediateLoadThreadResult&& workerResult) mutable {
            auto tileLoadResult = std::move(workerResult.tileLoadResult);
            auto meshes = std::move(workerResult.meshes);
            auto fabricMeshes = std::move(workerResult.fabricMeshes);
            const auto pModel = std::get_if<CesiumGltf::Model>(&tileLoadResult.contentKind);

            if (tilesetExists()) {
                setFabricTextures(*pModel, meshes, fabricMeshes);
            }

            return Cesium3DTilesSelection::TileLoadResultAndRenderResources{
                std::move(tileLoadResult),
                new TileLoadThreadResult{
                    std::move(meshes),
                    std::move(fabricMeshes),
                    transform,
                    hasImagery,
                },
            };
        });
}

void* FabricPrepareRenderResources::prepareInMainThread(Cesium3DTilesSelection::Tile& tile, void* pLoadThreadResult) {
    if (!pLoadThreadResult) {
        return nullptr;
    }

    // Wrap in a unique_ptr so that pLoadThreadResult gets freed when this function returns
    std::unique_ptr<TileLoadThreadResult> pTileLoadThreadResult{static_cast<TileLoadThreadResult*>(pLoadThreadResult)};

    const auto& meshes = pTileLoadThreadResult->meshes;
    auto& fabricMeshes = pTileLoadThreadResult->fabricMeshes;
    const auto hasImagery = pTileLoadThreadResult->hasImagery;
    const auto& tileTransform = pTileLoadThreadResult->tileTransform;

    const auto& content = tile.getContent();
    auto pRenderContent = content.getRenderContent();
    if (!pRenderContent) {
        return nullptr;
    }

    const auto& model = pRenderContent->getModel();

    if (tilesetExists()) {
        setFabricMeshes(model, meshes, fabricMeshes, hasImagery, *_tileset);
    }

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
        const auto pTileLoadThreadResult = static_cast<TileLoadThreadResult*>(pLoadThreadResult);
        freeFabricMeshes(pTileLoadThreadResult->fabricMeshes);
        delete pTileLoadThreadResult;
    }

    if (pMainThreadResult) {
        const auto pTileRenderResources = static_cast<TileRenderResources*>(pMainThreadResult);
        freeFabricMeshes(pTileRenderResources->fabricMeshes);
        delete pTileRenderResources;
    }
}

void* FabricPrepareRenderResources::prepareRasterInLoadThread(
    CesiumGltf::ImageCesium& image,
    [[maybe_unused]] const std::any& rendererOptions) {

    if (!tilesetExists()) {
        return nullptr;
    }

    auto texture = FabricResourceManager::getInstance().acquireTexture();
    texture->setImage(image);
    return new ImageryLoadThreadResult{texture};
}

void* FabricPrepareRenderResources::prepareRasterInMainThread(
    [[maybe_unused]] Cesium3DTilesSelection::RasterOverlayTile& rasterTile,
    void* pLoadThreadResult) {
    if (!pLoadThreadResult) {
        return nullptr;
    }

    // Wrap in a unique_ptr so that pLoadThreadResult gets freed when this function returns
    std::unique_ptr<ImageryLoadThreadResult> pImageryLoadThreadResult{
        static_cast<ImageryLoadThreadResult*>(pLoadThreadResult)};

    if (!tilesetExists()) {
        return nullptr;
    }

    auto texture = pImageryLoadThreadResult->texture;

    return new ImageryRenderResources{texture};
}

void FabricPrepareRenderResources::freeRaster(
    [[maybe_unused]] const Cesium3DTilesSelection::RasterOverlayTile& rasterTile,
    void* pLoadThreadResult,
    void* pMainThreadResult) noexcept {

    if (pLoadThreadResult) {
        const auto pImageryLoadThreadResult = static_cast<ImageryLoadThreadResult*>(pLoadThreadResult);
        const auto texture = pImageryLoadThreadResult->texture;
        FabricResourceManager::getInstance().releaseTexture(texture);
        delete pImageryLoadThreadResult;
    }

    if (pMainThreadResult) {
        const auto pImageryRenderResources = static_cast<ImageryRenderResources*>(pMainThreadResult);
        const auto texture = pImageryRenderResources->texture;
        FabricResourceManager::getInstance().releaseTexture(texture);
        delete pImageryRenderResources;
    }
}

void FabricPrepareRenderResources::attachRasterInMainThread(
    const Cesium3DTilesSelection::Tile& tile,
    int32_t overlayTextureCoordinateID,
    [[maybe_unused]] const Cesium3DTilesSelection::RasterOverlayTile& rasterTile,
    void* pMainThreadRendererResources,
    const glm::dvec2& translation,
    const glm::dvec2& scale) {

    auto pImageryRenderResources = static_cast<ImageryRenderResources*>(pMainThreadRendererResources);
    if (!pImageryRenderResources) {
        return;
    }

    if (!tilesetExists()) {
        return;
    }

    const auto texture = pImageryRenderResources->texture;

    auto& content = tile.getContent();
    auto pRenderContent = content.getRenderContent();
    if (!pRenderContent) {
        return;
    }

    auto pTileRenderResources = static_cast<TileRenderResources*>(pRenderContent->getRenderResources());
    if (!pTileRenderResources) {
        return;
    }

    for (const auto& mesh : pTileRenderResources->fabricMeshes) {
        auto& material = mesh.material;
        if (material != nullptr) {
            const auto textureInfo = TextureInfo{
                translation,
                0.0,
                scale,
                static_cast<uint64_t>(overlayTextureCoordinateID),
                CesiumGltf::Sampler::WrapS::CLAMP_TO_EDGE,
                CesiumGltf::Sampler::WrapT::CLAMP_TO_EDGE,
                false,
            };

            // Replace the original base color texture with the imagery
            material->setBaseColorTexture(texture, textureInfo);
        }
    }
}

void FabricPrepareRenderResources::detachRasterInMainThread(
    const Cesium3DTilesSelection::Tile& tile,
    [[maybe_unused]] int32_t overlayTextureCoordinateID,
    [[maybe_unused]] const Cesium3DTilesSelection::RasterOverlayTile& rasterTile,
    [[maybe_unused]] void* pMainThreadRendererResources) noexcept {

    auto& content = tile.getContent();
    auto pRenderContent = content.getRenderContent();
    if (!pRenderContent) {
        return;
    }

    auto pTileRenderResources = static_cast<TileRenderResources*>(pRenderContent->getRenderResources());
    if (!pTileRenderResources) {
        return;
    }

    if (!tilesetExists()) {
        return;
    }

    for (const auto& mesh : pTileRenderResources->fabricMeshes) {
        auto& material = mesh.material;
        const auto& baseColorTexture = mesh.baseColorTexture;
        const auto& materialInfo = mesh.materialInfo;

        if (material != nullptr) {
            if (baseColorTexture != nullptr && materialInfo.baseColorTexture.has_value()) {
                // Switch back to the original base color texture
                material->setBaseColorTexture(baseColorTexture, materialInfo.baseColorTexture.value());
            } else {
                material->clearBaseColorTexture();
            }
        }
    }
}

bool FabricPrepareRenderResources::tilesetExists() const {
    // When a tileset is deleted there's a short period between the prim being deleted and TfNotice notifying us about the change.
    // This function helps us know whether we should proceed with loading render resources.
    return _tileset != nullptr && UsdUtil::primExists(_tileset->getPath());
}

void FabricPrepareRenderResources::detachTileset() {
    _tileset = nullptr;
}

} // namespace cesium::omniverse
