#pragma once

#include "cesium/omniverse/ISubsystem.h"

#include <omni/fabric/IToken.h>

#include <filesystem>
#include <memory>

namespace omni::fabric {
class StageReaderWriter;
}

namespace cesium::omniverse {

class Context;
class FabricResourceManager;
class ILogger;
class KitLogger;
class KitSettings;
class SubsystemPrepareRendererResources;

class KitSubsystem final : public ISubsystem {
  public:
    KitSubsystem(Context* pContext, const std::filesystem::path& cesiumMdlPath);

    void clear() override;
    void applyDebugSettings(const DebugSettings& debugSettings) override;
    void releaseStage() override;
    void acquireStage(int64_t usdStageId) override;
    [[nodiscard]] SubsystemStatistics getStatistics() const override;
    void setTilesetTransform(int64_t tilesetId, const glm::dmat4& ecefToPrimWorldTransform) override;
    [[nodiscard]] std::string printStage() const override;
    [[nodiscard]] std::shared_ptr<SubsystemPrepareRendererResources>
    createPrepareRendererResources(Context* pContext, OmniTileset* pTileset) override;
    [[nodiscard]] bool materialHasCesiumNodes(const pxr::SdfPath& materialPath) const override;
    [[nodiscard]] bool
    isShaderConnectedToMaterial(const pxr::SdfPath& materialPath, const pxr::SdfPath& shaderPath) const override;
    void updateShaderInput(
        const pxr::SdfPath& materialPath,
        const pxr::SdfPath& shaderPath,
        const pxr::TfToken& property) const override;
    void updateRasterOverlayAlpha(
        const Cesium3DTilesSelection::Tileset& tileset,
        uint64_t rasterOverlayIndex,
        double alpha) const override;
    void updateDisplayColorAndOpacity(
        const Cesium3DTilesSelection::Tileset& tileset,
        const glm::dvec3& displayColor,
        double displayOpacity) const override;
    void updateShaderInput(
        const Cesium3DTilesSelection::Tileset& tileset,
        const pxr::SdfPath& shaderPath,
        const pxr::TfToken& attributeName) const override;
    void setContentVisiblity(void* pRenderResources, bool visible) const override;
    [[nodiscard]] std::shared_ptr<ILogger> getLogger() const override;
    [[nodiscard]] const ISettings& getSettings() const override;

    [[nodiscard]] const omni::fabric::Token& getCesiumMdlPathToken() const;
    [[nodiscard]] const FabricResourceManager& getFabricResourceManager() const;
    [[nodiscard]] FabricResourceManager& getFabricResourceManager();
    [[nodiscard]] omni::fabric::StageReaderWriter& getFabricStage() const;

  private:
    omni::fabric::Token _cesiumMdlPathToken;
    std::unique_ptr<FabricResourceManager> _pFabricResourceManager;
    std::shared_ptr<KitLogger> _pLogger;
    std::unique_ptr<KitSettings> _pSettings;
    std::unique_ptr<omni::fabric::StageReaderWriter> _pFabricStage;
};

} // namespace cesium::omniverse
