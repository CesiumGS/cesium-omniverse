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
class SubsystemPrepareRendererResources;

class KitSubsystem final : public ISubsystem {
  public:
    KitSubsystem(const std::filesystem::path& cesiumMdlPath);

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

    [[nodiscard]] const omni::fabric::Token& getCesiumMdlPathToken() const;
    [[nodiscard]] const FabricResourceManager& getFabricResourceManager() const;
    [[nodiscard]] FabricResourceManager& getFabricResourceManager();
    [[nodiscard]] omni::fabric::StageReaderWriter& getFabricStage() const;

  private:
    omni::fabric::Token _cesiumMdlPathToken;
    std::unique_ptr<FabricResourceManager> _pFabricResourceManager;
    std::unique_ptr<omni::fabric::StageReaderWriter> _pFabricStage;
};

} // namespace cesium::omniverse
