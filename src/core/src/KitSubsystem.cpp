#include "cesium/omniverse/KitSubsystem.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/DebugSettings.h"
#include "cesium/omniverse/FabricGeometry.h"
#include "cesium/omniverse/FabricMaterial.h"
#include "cesium/omniverse/FabricPrepareRenderResources.h"
#include "cesium/omniverse/FabricRenderResources.h"
#include "cesium/omniverse/FabricResourceManager.h"
#include "cesium/omniverse/FabricUtil.h"
#include "cesium/omniverse/SubsystemStatistics.h"

#include <Cesium3DTilesSelection/Tileset.h>
#include <omni/fabric/FabricUSD.h>

namespace cesium::omniverse {

namespace {
void forEachFabricMaterial(
    const Cesium3DTilesSelection::Tileset& tileset,
    const std::function<void(FabricMaterial& fabricMaterial)>& callback) {
    tileset.forEachLoadedTile([&callback](const Cesium3DTilesSelection::Tile& tile) {
        if (tile.getState() != Cesium3DTilesSelection::TileLoadState::Done) {
            return;
        }
        const auto& content = tile.getContent();
        const auto pRenderContent = content.getRenderContent();
        if (!pRenderContent) {
            return;
        }
        const auto pFabricRenderResources = static_cast<FabricRenderResources*>(pRenderContent->getRenderResources());
        if (!pFabricRenderResources) {
            return;
        }
        for (const auto& fabricMesh : pFabricRenderResources->fabricMeshes) {
            if (fabricMesh.pMaterial) {
                callback(*fabricMesh.pMaterial.get());
            }
        }
    });
}
} // namespace

KitSubsystem::KitSubsystem(const std::filesystem::path& cesiumMdlPath)
    : _cesiumMdlPathToken(
          omni::fabric::Token::createImmortal(cesiumMdlPath.lexically_normal().generic_string().c_str())) {}

void KitSubsystem::clear() {
    _pFabricResourceManager->clear();
}

void KitSubsystem::applyDebugSettings(const DebugSettings& debugSettings) {
    _pFabricResourceManager->setDisableMaterials(debugSettings.disableMaterials);
    _pFabricResourceManager->setDisableTextures(debugSettings.disableTextures);
    _pFabricResourceManager->setDisableGeometryPool(debugSettings.disableGeometryPool);
    _pFabricResourceManager->setDisableMaterialPool(debugSettings.disableMaterialPool);
    _pFabricResourceManager->setDisableTexturePool(debugSettings.disableTexturePool);
    _pFabricResourceManager->setGeometryPoolInitialCapacity(debugSettings.geometryPoolInitialCapacity);
    _pFabricResourceManager->setMaterialPoolInitialCapacity(debugSettings.materialPoolInitialCapacity);
    _pFabricResourceManager->setTexturePoolInitialCapacity(debugSettings.texturePoolInitialCapacity);
    _pFabricResourceManager->setDebugRandomColors(debugSettings.randomColors);
}

void KitSubsystem::releaseStage() {
    _pFabricStage = nullptr;
}

void KitSubsystem::acquireStage(int64_t usdStageId) {
    const auto iFabricStage = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    const auto fabricStageId = iFabricStage->get(omni::fabric::UsdStageId(static_cast<uint64_t>(usdStageId)));
    _pFabricStage = std::make_unique<omni::fabric::StageReaderWriter>(fabricStageId);
}

SubsystemStatistics KitSubsystem::getStatistics() const {
    return FabricUtil::getStatistics(getFabricStage());
}

void KitSubsystem::setTilesetTransform(int64_t tilesetId, const glm::dmat4& ecefToPrimWorldTransform) {
    FabricUtil::setTilesetTransform(getFabricStage(), tilesetId, ecefToPrimWorldTransform);
}

std::string KitSubsystem::printStage() const {
    return FabricUtil::printFabricStage(getFabricStage());
}

std::shared_ptr<SubsystemPrepareRendererResources>
KitSubsystem::createPrepareRendererResources(Context* pContext, OmniTileset* pTileset) {
    return std::make_shared<FabricPrepareRenderResources>(pContext, this, pTileset);
}

bool KitSubsystem::materialHasCesiumNodes(const pxr::SdfPath& materialPath) const {
    const auto materialPathFabric = FabricUtil::toFabricPath(materialPath);
    return FabricUtil::materialHasCesiumNodes(getFabricStage(), materialPathFabric);
}

bool KitSubsystem::isShaderConnectedToMaterial(const pxr::SdfPath& materialPath, const pxr::SdfPath& shaderPath) const {
    const auto materialPathFabric = FabricUtil::toFabricPath(materialPath);
    const auto shaderPathFabric = FabricUtil::toFabricPath(shaderPath);
    return FabricUtil::isShaderConnectedToMaterial(getFabricStage(), materialPathFabric, shaderPathFabric);
}

void KitSubsystem::updateShaderInput(
    const pxr::SdfPath& materialPath,
    const pxr::SdfPath& shaderPath,
    const pxr::TfToken& property) const {
    getFabricResourceManager().updateShaderInput(materialPath, shaderPath, property);
}

void KitSubsystem::updateRasterOverlayAlpha(
    const Cesium3DTilesSelection::Tileset& tileset,
    uint64_t rasterOverlayIndex,
    double alpha) const {
    forEachFabricMaterial(tileset, [rasterOverlayIndex, alpha](FabricMaterial& fabricMaterial) {
        fabricMaterial.setRasterOverlayAlpha(rasterOverlayIndex, alpha);
    });
}

void KitSubsystem::updateDisplayColorAndOpacity(
    const Cesium3DTilesSelection::Tileset& tileset,
    const glm::dvec3& displayColor,
    double displayOpacity) const {
    forEachFabricMaterial(tileset, [&displayColor, &displayOpacity](FabricMaterial& fabricMaterial) {
        fabricMaterial.setDisplayColorAndOpacity(displayColor, displayOpacity);
    });
}

void KitSubsystem::updateShaderInput(
    const Cesium3DTilesSelection::Tileset& tileset,
    const pxr::SdfPath& shaderPath,
    const pxr::TfToken& attributeName) const {
    forEachFabricMaterial(tileset, [&shaderPath, &attributeName](FabricMaterial& fabricMaterial) {
        fabricMaterial.updateShaderInput(
            FabricUtil::toFabricPath(shaderPath), FabricUtil::toFabricToken(attributeName));
    });
}

void KitSubsystem::setContentVisiblity(void* pRenderResources, bool visible) const {
    assert(pRenderResources);
    const auto& fabricRenderResources = *static_cast<const FabricRenderResources*>(pRenderResources);

    for (const auto& fabricMesh : fabricRenderResources.fabricMeshes) {
        fabricMesh.pGeometry->setVisibility(visible);
    }
}

const omni::fabric::Token& KitSubsystem::getCesiumMdlPathToken() const {
    return _cesiumMdlPathToken;
}

const FabricResourceManager& KitSubsystem::getFabricResourceManager() const {
    return *_pFabricResourceManager.get();
}

FabricResourceManager& KitSubsystem::getFabricResourceManager() {
    return *_pFabricResourceManager.get();
}

omni::fabric::StageReaderWriter& KitSubsystem::getFabricStage() const {
    return *_pFabricStage.get();
}

} // namespace cesium::omniverse
