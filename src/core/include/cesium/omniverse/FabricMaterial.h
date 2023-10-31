#pragma once

#include "cesium/omniverse/FabricMaterialDefinition.h"
#include "cesium/omniverse/GltfUtil.h"

#include <omni/fabric/IPath.h>
#include <omni/fabric/Type.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/sdf/path.h>

namespace omni::ui {
class DynamicTextureProvider;
}

namespace cesium::omniverse {

class FabricMaterial {
  public:
    FabricMaterial(
        const omni::fabric::Path& path,
        const FabricMaterialDefinition& materialDefinition,
        const pxr::TfToken& defaultTextureAssetPathToken,
        const pxr::TfToken& defaultTransparentTextureAssetPathToken,
        bool debugRandomColors,
        long stageId);
    ~FabricMaterial();

    void setMaterial(int64_t tilesetId, const MaterialInfo& materialInfo);
    void setBaseColorTexture(
        const pxr::TfToken& textureAssetPathToken,
        const TextureInfo& textureInfo,
        uint64_t texcoordIndex);
    void setImageryLayer(
        const pxr::TfToken& textureAssetPathToken,
        const TextureInfo& textureInfo,
        uint64_t texcoordIndex,
        uint64_t imageryLayerIndex,
        double alpha);
    void setImageryLayerAlpha(uint64_t imageryLayerIndex, double alpha);
    void updateShaderInput(const omni::fabric::Path& shaderPath, const omni::fabric::Token& attributeName);

    void clearMaterial();
    void clearBaseColorTexture();
    void clearImageryLayer(uint64_t imageryLayerIndex);
    void clearImageryLayers();

    void setActive(bool active);

    [[nodiscard]] const omni::fabric::Path& getPath() const;
    [[nodiscard]] const FabricMaterialDefinition& getMaterialDefinition() const;

  private:
    void initialize();
    void initializeFromExistingMaterial(const omni::fabric::Path& path);

    void createMaterial(const omni::fabric::Path& materialPath);
    void createShader(const omni::fabric::Path& shaderPath, const omni::fabric::Path& materialPath);
    void createTextureCommon(
        const omni::fabric::Path& texturePath,
        const omni::fabric::Path& shaderPath,
        const omni::fabric::Token& shaderInput,
        const omni::fabric::Token& subIdentifier,
        const std::vector<std::pair<omni::fabric::Type, omni::fabric::Token>>& additionalAttributes = {});
    void createTexture(
        const omni::fabric::Path& texturePath,
        const omni::fabric::Path& shaderPath,
        const omni::fabric::Token& shaderInput);
    void createImageryLayer(
        const omni::fabric::Path& imageryLayerPath,
        const omni::fabric::Path& shaderPath,
        const omni::fabric::Token& shaderInput);
    void createImageryLayerResolver(
        const omni::fabric::Path& imageryLayerResolverPath,
        const omni::fabric::Path& shaderPath,
        const omni::fabric::Token& shaderInput,
        uint64_t textureCount);
    void reset();
    void setShaderValues(const omni::fabric::Path& shaderPath, const MaterialInfo& materialInfo);
    void setTextureValuesCommon(
        const omni::fabric::Path& texturePath,
        const pxr::TfToken& textureAssetPathToken,
        const TextureInfo& textureInfo,
        uint64_t texcoordIndex);
    void setTextureValues(
        const omni::fabric::Path& texturePath,
        const pxr::TfToken& textureAssetPathToken,
        const TextureInfo& textureInfo,
        uint64_t texcoordIndex);
    void setImageryLayerValues(
        const omni::fabric::Path& imageryLayerPath,
        const pxr::TfToken& textureAssetPathToken,
        const TextureInfo& textureInfo,
        uint64_t texcoordIndex,
        double alpha);
    void setImageryLayerAlphaValue(const omni::fabric::Path& imageryLayerPath, double alpha);

    bool stageDestroyed();

    omni::fabric::Path _materialPath;
    const FabricMaterialDefinition _materialDefinition;
    const pxr::TfToken _defaultTextureAssetPathToken;
    const pxr::TfToken _defaultTransparentTextureAssetPathToken;
    const bool _debugRandomColors;
    const long _stageId;

    std::vector<omni::fabric::Path> _shaderPaths;
    std::vector<omni::fabric::Path> _baseColorTexturePaths;
    std::vector<std::vector<omni::fabric::Path>> _imageryLayerPaths;

    std::vector<omni::fabric::Path> _allPaths;
};

} // namespace cesium::omniverse
