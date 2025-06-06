#include "cesium/omniverse/FabricPrepareRenderResources.h"

#include "cesium/omniverse/AssetRegistry.h"
#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/CppUtil.h"
#include "cesium/omniverse/FabricFeaturesInfo.h"
#include "cesium/omniverse/FabricFeaturesUtil.h"
#include "cesium/omniverse/FabricGeometry.h"
#include "cesium/omniverse/FabricMaterial.h"
#include "cesium/omniverse/FabricMesh.h"
#include "cesium/omniverse/FabricRasterOverlaysInfo.h"
#include "cesium/omniverse/FabricRenderResources.h"
#include "cesium/omniverse/FabricResourceManager.h"
#include "cesium/omniverse/FabricTexture.h"
#include "cesium/omniverse/FabricTextureData.h"
#include "cesium/omniverse/FabricUtil.h"
#include "cesium/omniverse/GltfUtil.h"
#include "cesium/omniverse/MetadataUtil.h"
#include "cesium/omniverse/OmniRasterOverlay.h"
#include "cesium/omniverse/OmniTileset.h"
#include "cesium/omniverse/UsdUtil.h"

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <Cesium3DTilesSelection/Tile.h>
#include <Cesium3DTilesSelection/Tileset.h>
#include <CesiumAsync/AsyncSystem.h>
#include <CesiumGltfContent/GltfUtilities.h>
#include <omni/fabric/FabricUSD.h>
#include <omni/ui/ImageProvider/DynamicTextureProvider.h>

namespace cesium::omniverse {

namespace {

struct RasterOverlayLoadThreadResult {
    std::shared_ptr<FabricTexture> pTexture;
};

struct RasterOverlayRenderResources {
    std::shared_ptr<FabricTexture> pTexture;
};

struct LoadingMesh {
    const glm::dmat4 gltfLocalToEcefTransform;
    const uint64_t gltfMeshIndex;
    const uint64_t gltfPrimitiveIndex;
};

struct TileLoadThreadResult {
    std::vector<LoadingMesh> loadingMeshes;
    std::vector<FabricMesh> fabricMeshes;
};

uint64_t getFeatureIdTextureCount(const FabricFeaturesInfo& fabricFeaturesInfo) {
    return CppUtil::countIf(fabricFeaturesInfo.featureIds, [](const auto& featureId) {
        return std::holds_alternative<FabricTextureInfo>(featureId.featureIdStorage);
    });
}

std::vector<LoadingMesh> getLoadingMeshes(const glm::dmat4& tileToEcefTransform, const CesiumGltf::Model& model) {
    CESIUM_TRACE("FabricPrepareRenderResources::getLoadingMeshes");

    auto gltfWorldToTileTransform = glm::dmat4(1.0);
    gltfWorldToTileTransform = CesiumGltfContent::GltfUtilities::applyRtcCenter(model, gltfWorldToTileTransform);
    gltfWorldToTileTransform =
        CesiumGltfContent::GltfUtilities::applyGltfUpAxisTransform(model, gltfWorldToTileTransform);

    std::vector<LoadingMesh> loadingMeshes;

    model.forEachPrimitiveInScene(
        -1,
        [&tileToEcefTransform, &gltfWorldToTileTransform, &loadingMeshes](
            const CesiumGltf::Model& gltf,
            [[maybe_unused]] const CesiumGltf::Node& node,
            const CesiumGltf::Mesh& mesh,
            const CesiumGltf::MeshPrimitive& primitive,
            const glm::dmat4& gltfLocalToWorldTransform) {
            const auto gltfMeshIndex = CppUtil::getIndexFromRef(gltf.meshes, mesh);
            const auto gltfPrimitiveIndex = CppUtil::getIndexFromRef(mesh.primitives, primitive);
            const auto gltfLocalToEcefTransform =
                tileToEcefTransform * gltfWorldToTileTransform * gltfLocalToWorldTransform;
            // In C++ 20 this can be emplace_back without the {}
            loadingMeshes.push_back({
                gltfLocalToEcefTransform,
                gltfMeshIndex,
                gltfPrimitiveIndex,
            });
        });

    return loadingMeshes;
}

std::vector<FabricMesh> acquireFabricMeshes(
    Context& context,
    const CesiumGltf::Model& model,
    const std::vector<LoadingMesh>& loadingMeshes,
    const FabricRasterOverlaysInfo& rasterOverlaysInfo,
    const OmniTileset& tileset) {
    CESIUM_TRACE("FabricPrepareRenderResources::acquireFabricMeshes");
    std::vector<FabricMesh> fabricMeshes;
    fabricMeshes.reserve(loadingMeshes.size());

    auto& fabricResourceManager = context.getFabricResourceManager();
    const auto tilesetMaterialPath = tileset.getMaterialPath();

    for (const auto& loadingMesh : loadingMeshes) {
        auto& fabricMesh = fabricMeshes.emplace_back();

        const auto& primitive = model.meshes[loadingMesh.gltfMeshIndex].primitives[loadingMesh.gltfPrimitiveIndex];

        const auto materialInfo = GltfUtil::getMaterialInfo(model, primitive);
        const auto featuresInfo = GltfUtil::getFeaturesInfo(model, primitive);

        const auto shouldAcquireMaterial = fabricResourceManager.shouldAcquireMaterial(
            primitive, rasterOverlaysInfo.overlayRenderMethods.size() > 0, tilesetMaterialPath);

        fabricMesh.materialInfo = materialInfo;
        fabricMesh.featuresInfo = featuresInfo;

        fabricMesh.pGeometry =
            fabricResourceManager.acquireGeometry(model, primitive, featuresInfo, tileset.getSmoothNormals());

        if (shouldAcquireMaterial) {
            fabricMesh.pMaterial = fabricResourceManager.acquireMaterial(
                model,
                primitive,
                materialInfo,
                featuresInfo,
                rasterOverlaysInfo,
                tileset.getTilesetId(),
                tilesetMaterialPath);
        }

        if (materialInfo.baseColorTexture.has_value()) {
            fabricMesh.pBaseColorTexture = fabricResourceManager.acquireTexture();
        }

        const auto featureIdTextureCount = getFeatureIdTextureCount(featuresInfo);
        fabricMesh.featureIdTextures.reserve(featureIdTextureCount);
        for (uint64_t i = 0; i < featureIdTextureCount; ++i) {
            fabricMesh.featureIdTextures.push_back(fabricResourceManager.acquireTexture());
        }

        const auto propertyTextureCount = MetadataUtil::getPropertyTextureImages(context, model, primitive).size();
        fabricMesh.propertyTextures.reserve(propertyTextureCount);
        for (uint64_t i = 0; i < propertyTextureCount; ++i) {
            fabricMesh.propertyTextures.push_back(fabricResourceManager.acquireTexture());
        }

        const auto propertyTableTextureCount = MetadataUtil::getPropertyTableTextureCount(context, model, primitive);
        fabricMesh.propertyTableTextures.reserve(propertyTableTextureCount);
        for (uint64_t i = 0; i < propertyTableTextureCount; ++i) {
            fabricMesh.propertyTableTextures.push_back(fabricResourceManager.acquireTexture());
        }

        // Map glTF texcoord set index to primvar st index
        const auto texcoordSetIndexes = GltfUtil::getTexcoordSetIndexes(model, primitive);
        const auto rasterOverlayTexcoordSetIndexes = GltfUtil::getRasterOverlayTexcoordSetIndexes(model, primitive);

        uint64_t primvarStIndex = 0;
        for (const auto gltfSetIndex : texcoordSetIndexes) {
            fabricMesh.texcoordIndexMapping[gltfSetIndex] = primvarStIndex++;
        }
        for (const auto gltfSetIndex : rasterOverlayTexcoordSetIndexes) {
            fabricMesh.rasterOverlayTexcoordIndexMapping[gltfSetIndex] = primvarStIndex++;
        }

        // Map feature id types to set indexes
        fabricMesh.featureIdIndexSetIndexMapping =
            FabricFeaturesUtil::getSetIndexMapping(featuresInfo, FabricFeatureIdType::INDEX);
        fabricMesh.featureIdAttributeSetIndexMapping =
            FabricFeaturesUtil::getSetIndexMapping(featuresInfo, FabricFeatureIdType::ATTRIBUTE);
        fabricMesh.featureIdTextureSetIndexMapping =
            FabricFeaturesUtil::getSetIndexMapping(featuresInfo, FabricFeatureIdType::TEXTURE);

        // Map glTF texture index to property texture (FabricTexture) index
        fabricMesh.propertyTextureIndexMapping =
            MetadataUtil::getPropertyTextureIndexMapping(context, model, primitive);
    }

    return fabricMeshes;
}

void setFabricTextures(
    const Context& context,
    const CesiumGltf::Model& model,
    const std::vector<LoadingMesh>& loadingMeshes,
    std::vector<FabricMesh>& fabricMeshes) {
    CESIUM_TRACE("FabricPrepareRenderResources::setFabricTextures");
    for (uint64_t i = 0; i < loadingMeshes.size(); ++i) {
        const auto& loadingMesh = loadingMeshes[i];
        const auto& primitive = model.meshes[loadingMesh.gltfMeshIndex].primitives[loadingMesh.gltfPrimitiveIndex];
        auto& fabricMesh = fabricMeshes[i];

        if (fabricMesh.pBaseColorTexture) {
            const auto pBaseColorTextureImage = GltfUtil::getBaseColorTextureImage(model, primitive);
            if (!pBaseColorTextureImage || context.getFabricResourceManager().getDisableTextures()) {
                fabricMesh.pBaseColorTexture->setBytes(
                    {std::byte(255), std::byte(255), std::byte(255), std::byte(255)}, 1, 1, carb::Format::eRGBA8_SRGB);
            } else {
                fabricMesh.pBaseColorTexture->setImage(*pBaseColorTextureImage, TransferFunction::SRGB);
            }
        }

        const auto featureIdTextureCount = fabricMesh.featureIdTextures.size();
        for (uint64_t j = 0; j < featureIdTextureCount; ++j) {
            const auto featureIdSetIndex = fabricMesh.featureIdTextureSetIndexMapping[j];
            const auto pFeatureIdTextureImage = GltfUtil::getFeatureIdTextureImage(model, primitive, featureIdSetIndex);
            if (!pFeatureIdTextureImage) {
                fabricMesh.featureIdTextures[j]->setBytes(
                    {std::byte(0), std::byte(0), std::byte(0), std::byte(0)}, 1, 1, carb::Format::eRGBA8_SRGB);
            } else {
                fabricMesh.featureIdTextures[j]->setImage(*pFeatureIdTextureImage, TransferFunction::LINEAR);
            }
        }

        const auto propertyTextureImages = MetadataUtil::getPropertyTextureImages(context, model, primitive);
        const auto propertyTextureCount = fabricMesh.propertyTextures.size();
        for (uint64_t j = 0; j < propertyTextureCount; ++j) {
            fabricMesh.propertyTextures[j]->setImage(*propertyTextureImages[j], TransferFunction::LINEAR);
        }

        const auto propertyTableTextures = MetadataUtil::encodePropertyTables(context, model, primitive);
        const auto propertyTableTextureCount = fabricMesh.propertyTableTextures.size();
        for (uint64_t j = 0; j < propertyTableTextureCount; ++j) {
            const auto& texture = propertyTableTextures[j];
            fabricMesh.propertyTableTextures[j]->setBytes(texture.bytes, texture.width, texture.height, texture.format);
        }
    }
}

void setFabricMeshes(
    const Context& context,
    const CesiumGltf::Model& model,
    const std::vector<LoadingMesh>& loadingMeshes,
    std::vector<FabricMesh>& fabricMeshes,
    const OmniTileset& tileset) {
    CESIUM_TRACE("FabricPrepareRenderResources::setFabricMeshes");

    const auto& tilesetMaterialPath = tileset.getMaterialPath();
    const auto displayColor = tileset.getDisplayColor();
    const auto displayOpacity = tileset.getDisplayOpacity();

    const auto ecefToPrimWorldTransform =
        UsdUtil::computeEcefToPrimWorldTransform(context, tileset.getResolvedGeoreferencePath(), tileset.getPath());

    const auto tilesetId = tileset.getTilesetId();
    const auto smoothNormals = tileset.getSmoothNormals();
    const auto pointSize = tileset.getPointSize();

    for (uint64_t i = 0; i < loadingMeshes.size(); ++i) {
        const auto& loadingMesh = loadingMeshes[i];
        const auto& primitive = model.meshes[loadingMesh.gltfMeshIndex].primitives[loadingMesh.gltfPrimitiveIndex];

        const auto& fabricMesh = fabricMeshes[i];
        const auto pGeometry = fabricMesh.pGeometry;
        const auto pMaterial = fabricMesh.pMaterial;

        pGeometry->setGeometry(
            tilesetId,
            ecefToPrimWorldTransform,
            loadingMesh.gltfLocalToEcefTransform,
            model,
            primitive,
            fabricMesh.materialInfo,
            smoothNormals,
            pointSize,
            fabricMesh.texcoordIndexMapping,
            fabricMesh.rasterOverlayTexcoordIndexMapping);

        if (pMaterial) {
            pMaterial->setMaterial(
                model,
                primitive,
                tilesetId,
                fabricMesh.materialInfo,
                fabricMesh.featuresInfo,
                fabricMesh.pBaseColorTexture.get(),
                fabricMesh.featureIdTextures,
                fabricMesh.propertyTextures,
                fabricMesh.propertyTableTextures,
                displayColor,
                displayOpacity,
                fabricMesh.texcoordIndexMapping,
                fabricMesh.featureIdIndexSetIndexMapping,
                fabricMesh.featureIdAttributeSetIndexMapping,
                fabricMesh.featureIdTextureSetIndexMapping,
                fabricMesh.propertyTextureIndexMapping);

            pGeometry->setMaterial(pMaterial->getPath());
        } else if (!tilesetMaterialPath.IsEmpty()) {
            pGeometry->setMaterial(FabricUtil::toFabricPath(tilesetMaterialPath));
        }
    }
}

void freeFabricMeshes(Context& context, const std::vector<FabricMesh>& fabricMeshes) {
    auto& fabricResourceManager = context.getFabricResourceManager();

    for (const auto& fabricMesh : fabricMeshes) {
        if (fabricMesh.pGeometry) {
            fabricResourceManager.releaseGeometry(fabricMesh.pGeometry);
        }

        if (fabricMesh.pMaterial) {
            fabricResourceManager.releaseMaterial(fabricMesh.pMaterial);
        }

        if (fabricMesh.pBaseColorTexture) {
            fabricResourceManager.releaseTexture(fabricMesh.pBaseColorTexture);
        }

        for (const auto& pFeatureIdTexture : fabricMesh.featureIdTextures) {
            fabricResourceManager.releaseTexture(pFeatureIdTexture);
        }

        for (const auto& pPropertyTexture : fabricMesh.propertyTextures) {
            fabricResourceManager.releaseTexture(pPropertyTexture);
        }

        for (const auto& pPropertyTableTexture : fabricMesh.propertyTableTextures) {
            fabricResourceManager.releaseTexture(pPropertyTableTexture);
        }
    }
}

} // namespace

FabricPrepareRenderResources::FabricPrepareRenderResources(Context* pContext, OmniTileset* pTileset)
    : _pContext(pContext)
    , _pTileset(pTileset) {}

CesiumAsync::Future<Cesium3DTilesSelection::TileLoadResultAndRenderResources>
FabricPrepareRenderResources::prepareInLoadThread(
    const CesiumAsync::AsyncSystem& asyncSystem,
    Cesium3DTilesSelection::TileLoadResult&& tileLoadResult,
    const glm::dmat4& tileToEcefTransform,
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

    // We don't know how many raster overlays actually overlap this tile until attachRasterInMainThread is called
    // but at least we have an upper bound. Unused texture slots are initialized with a 1x1 transparent pixel so
    // blending still works.
    const auto overlapsRasterOverlay = tileLoadResult.rasterOverlayDetails.has_value();

    FabricRasterOverlaysInfo rasterOverlaysInfo;

    if (overlapsRasterOverlay) {
        for (const auto& rasterOverlayPath : _pTileset->getRasterOverlayPaths()) {
            const auto& pRasterOverlay = _pContext->getAssetRegistry().getRasterOverlay(rasterOverlayPath);
            const auto overlayRenderMethod =
                pRasterOverlay ? pRasterOverlay->getOverlayRenderMethod() : FabricOverlayRenderMethod::OVERLAY;
            rasterOverlaysInfo.overlayRenderMethods.push_back(overlayRenderMethod);
        }
    }

    auto loadingMeshes = getLoadingMeshes(tileToEcefTransform, *pModel);

    struct IntermediateLoadThreadResult {
        Cesium3DTilesSelection::TileLoadResult tileLoadResult;
        std::vector<LoadingMesh> loadingMeshes;
        std::vector<FabricMesh> fabricMeshes;
    };

    return asyncSystem
        .runInMainThread([this,
                          rasterOverlaysInfo = std::move(rasterOverlaysInfo),
                          loadingMeshes = std::move(loadingMeshes),
                          tileLoadResult = std::move(tileLoadResult)]() mutable {
            if (!tilesetExists()) {
                return IntermediateLoadThreadResult{
                    std::move(tileLoadResult),
                    {},
                    {},
                };
            }

            const auto pModel = std::get_if<CesiumGltf::Model>(&tileLoadResult.contentKind);
            auto fabricMeshes = acquireFabricMeshes(*_pContext, *pModel, loadingMeshes, rasterOverlaysInfo, *_pTileset);
            return IntermediateLoadThreadResult{
                std::move(tileLoadResult),
                std::move(loadingMeshes),
                std::move(fabricMeshes),
            };
        })
        // Previously this was thenInWorkerThread, but as of Kit 106.5 it's not safe to set texture data off the main thread
        // See https://github.com/CesiumGS/cesium-omniverse/pull/721#issuecomment-2697969830
        .thenImmediately([this](IntermediateLoadThreadResult&& workerResult) mutable {
            auto tileLoadResult = std::move(workerResult.tileLoadResult);
            auto loadingMeshes = std::move(workerResult.loadingMeshes);
            auto fabricMeshes = std::move(workerResult.fabricMeshes);
            const auto pModel = std::get_if<CesiumGltf::Model>(&tileLoadResult.contentKind);

            if (tilesetExists()) {
                setFabricTextures(*_pContext, *pModel, loadingMeshes, fabricMeshes);
            }

            return Cesium3DTilesSelection::TileLoadResultAndRenderResources{
                std::move(tileLoadResult),
                new TileLoadThreadResult{
                    std::move(loadingMeshes),
                    std::move(fabricMeshes),
                },
            };
        });
}

void* FabricPrepareRenderResources::prepareInMainThread(Cesium3DTilesSelection::Tile& tile, void* pLoadThreadResult) {
    if (!pLoadThreadResult) {
        return nullptr;
    }

    // Wrap in a unique_ptr so that pLoadThreadResult gets freed when this function returns
    std::unique_ptr<TileLoadThreadResult> pTileLoadThreadResult(static_cast<TileLoadThreadResult*>(pLoadThreadResult));

    const auto& loadingMeshes = pTileLoadThreadResult->loadingMeshes;
    auto& fabricMeshes = pTileLoadThreadResult->fabricMeshes;

    const auto& content = tile.getContent();
    auto pRenderContent = content.getRenderContent();
    if (!pRenderContent) {
        return nullptr;
    }

    const auto& model = pRenderContent->getModel();

    if (tilesetExists()) {
        setFabricMeshes(*_pContext, model, loadingMeshes, fabricMeshes, *_pTileset);
    }

    return new FabricRenderResources{
        std::move(fabricMeshes),
    };
}

void FabricPrepareRenderResources::free(
    [[maybe_unused]] Cesium3DTilesSelection::Tile& tile,
    void* pLoadThreadResult,
    void* pMainThreadResult) noexcept {
    if (pLoadThreadResult) {
        const auto pTileLoadThreadResult = static_cast<TileLoadThreadResult*>(pLoadThreadResult);
        freeFabricMeshes(*_pContext, pTileLoadThreadResult->fabricMeshes);
        delete pTileLoadThreadResult;
    }

    if (pMainThreadResult) {
        const auto pFabricRenderResources = static_cast<FabricRenderResources*>(pMainThreadResult);
        freeFabricMeshes(*_pContext, pFabricRenderResources->fabricMeshes);
        delete pFabricRenderResources;
    }
}

void* FabricPrepareRenderResources::prepareRasterInLoadThread(
    [[maybe_unused]] CesiumGltf::ImageCesium& image,
    [[maybe_unused]] const std::any& rendererOptions) {
    return nullptr;
}

void* FabricPrepareRenderResources::prepareRasterInMainThread(
    [[maybe_unused]] CesiumRasterOverlays::RasterOverlayTile& rasterTile,
    [[maybe_unused]] void* pLoadThreadResult) {
    if (!tilesetExists()) {
        return nullptr;
    }

    // Previously this was in prepareRasterInLoadThread, but as of Kit 106.5 it's not safe to set texture data off the main thread
    // See https://github.com/CesiumGS/cesium-omniverse/pull/721#issuecomment-2697969830
    const auto pTexture = _pContext->getFabricResourceManager().acquireTexture();
    pTexture->setImage(rasterTile.getImage(), TransferFunction::SRGB);
    return new RasterOverlayRenderResources{pTexture};
}

void FabricPrepareRenderResources::freeRaster(
    [[maybe_unused]] const CesiumRasterOverlays::RasterOverlayTile& rasterTile,
    void* pLoadThreadResult,
    void* pMainThreadResult) noexcept {

    if (pLoadThreadResult) {
        const auto pRasterOverlayLoadThreadResult = static_cast<RasterOverlayLoadThreadResult*>(pLoadThreadResult);
        _pContext->getFabricResourceManager().releaseTexture(pRasterOverlayLoadThreadResult->pTexture);
        delete pRasterOverlayLoadThreadResult;
    }

    if (pMainThreadResult) {
        const auto pRasterOverlayRenderResources = static_cast<RasterOverlayRenderResources*>(pMainThreadResult);
        _pContext->getFabricResourceManager().releaseTexture(pRasterOverlayRenderResources->pTexture);
        delete pRasterOverlayRenderResources;
    }
}

void FabricPrepareRenderResources::attachRasterInMainThread(
    const Cesium3DTilesSelection::Tile& tile,
    int32_t overlayTextureCoordinateID,
    [[maybe_unused]] const CesiumRasterOverlays::RasterOverlayTile& rasterTile,
    void* pMainThreadRendererResources,
    const glm::dvec2& translation,
    const glm::dvec2& scale) {

    auto pRasterOverlayRenderResources = static_cast<RasterOverlayRenderResources*>(pMainThreadRendererResources);
    if (!pRasterOverlayRenderResources) {
        return;
    }

    if (!tilesetExists()) {
        return;
    }

    const auto pTexture = pRasterOverlayRenderResources->pTexture.get();

    const auto& content = tile.getContent();
    const auto pRenderContent = content.getRenderContent();
    if (!pRenderContent) {
        return;
    }

    const auto pFabricRenderResources = static_cast<FabricRenderResources*>(pRenderContent->getRenderResources());
    if (!pFabricRenderResources) {
        return;
    }

    const auto rasterOverlayPath = _pTileset->getRasterOverlayPathIfExists(rasterTile.getOverlay());

    if (rasterOverlayPath.IsEmpty()) {
        return;
    }

    const auto rasterOverlayPaths = _pTileset->getRasterOverlayPaths();
    const auto rasterOverlayIndex = CppUtil::indexOf(_pTileset->getRasterOverlayPaths(), rasterOverlayPath);

    if (rasterOverlayIndex == rasterOverlayPaths.size()) {
        return;
    }

    const auto pRasterOverlay = _pContext->getAssetRegistry().getRasterOverlay(rasterOverlayPath);

    if (!pRasterOverlay) {
        return;
    }

    const auto alpha = glm::clamp(pRasterOverlay->getAlpha(), 0.0, 1.0);

    for (const auto& fabricMesh : pFabricRenderResources->fabricMeshes) {
        const auto pMaterial = fabricMesh.pMaterial;
        if (pMaterial) {
            const auto gltfSetIndex = static_cast<uint64_t>(overlayTextureCoordinateID);
            const auto textureInfo = FabricTextureInfo{
                translation,
                0.0,
                scale,
                gltfSetIndex,
                CesiumGltf::Sampler::WrapS::CLAMP_TO_EDGE,
                CesiumGltf::Sampler::WrapT::CLAMP_TO_EDGE,
                false,
                {},
            };
            pMaterial->setRasterOverlay(
                pTexture, textureInfo, rasterOverlayIndex, alpha, fabricMesh.rasterOverlayTexcoordIndexMapping);
        }
    }
}

void FabricPrepareRenderResources::detachRasterInMainThread(
    const Cesium3DTilesSelection::Tile& tile,
    [[maybe_unused]] int32_t overlayTextureCoordinateID,
    const CesiumRasterOverlays::RasterOverlayTile& rasterTile,
    [[maybe_unused]] void* pMainThreadRendererResources) noexcept {

    const auto& content = tile.getContent();
    const auto pRenderContent = content.getRenderContent();
    if (!pRenderContent) {
        return;
    }

    const auto pFabricRenderResources = static_cast<FabricRenderResources*>(pRenderContent->getRenderResources());
    if (!pFabricRenderResources) {
        return;
    }

    if (!tilesetExists()) {
        return;
    }

    const auto rasterOverlayPath = _pTileset->getRasterOverlayPathIfExists(rasterTile.getOverlay());

    if (rasterOverlayPath.IsEmpty()) {
        return;
    }

    const auto rasterOverlayPaths = _pTileset->getRasterOverlayPaths();
    const auto rasterOverlayIndex = CppUtil::indexOf(rasterOverlayPaths, rasterOverlayPath);

    if (rasterOverlayIndex == rasterOverlayPaths.size()) {
        return;
    }

    for (const auto& fabricMesh : pFabricRenderResources->fabricMeshes) {
        const auto pMaterial = fabricMesh.pMaterial;
        if (pMaterial) {
            pMaterial->clearRasterOverlay(rasterOverlayIndex);
        }
    }
}

bool FabricPrepareRenderResources::tilesetExists() const {
    // When a tileset is deleted there's a short period between the prim being deleted and TfNotice notifying us about the change.
    // This function helps us know whether we should proceed with loading render resources.
    return _pTileset && _pContext->hasUsdStage() && UsdUtil::primExists(_pContext->getUsdStage(), _pTileset->getPath());
}

void FabricPrepareRenderResources::detachTileset() {
    _pTileset = nullptr;
}

} // namespace cesium::omniverse
