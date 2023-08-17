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
        long stageId);
    ~FabricMaterial();

    void setMaterial(int64_t tilesetId, const MaterialInfo& materialInfo);
    void setBaseColorTexture(const pxr::TfToken& textureAssetPathToken, const TextureInfo& textureInfo);

    void clearMaterial();
    void clearBaseColorTexture();

    void setActive(bool active);

    [[nodiscard]] const omni::fabric::Path& getPath() const;
    [[nodiscard]] const FabricMaterialDefinition& getMaterialDefinition() const;

  private:
    void initialize();

    void createMaterial(const omni::fabric::Path& materialPath);
    void createShader(const omni::fabric::Path& shaderPath, const omni::fabric::Path& materialPath);
    void createTexture(
        const omni::fabric::Path& texturePath,
        const omni::fabric::Path& shaderPath,
        const omni::fabric::Token& shaderInput);

    void reset();
    void setShaderValues(const omni::fabric::Path& shaderPath, const MaterialInfo& materialInfo);
    void setTextureValues(
        const omni::fabric::Path& texturePath,
        const pxr::TfToken& textureAssetPathToken,
        const TextureInfo& textureInfo);
    void setTilesetId(int64_t tilesetId);
    bool stageDestroyed();

    omni::fabric::Path _materialPath;
    const FabricMaterialDefinition _materialDefinition;
    const pxr::TfToken _defaultTextureAssetPathToken;
    const long _stageId;

    omni::fabric::Path _shaderPath;
    omni::fabric::Path _baseColorTexturePath;
};

} // namespace cesium::omniverse
