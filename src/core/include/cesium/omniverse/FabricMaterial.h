#pragma once

#include "cesium/omniverse/FabricMaterialDefinition.h"
#include "cesium/omniverse/GltfUtil.h"

#include <omni/fabric/IPath.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/sdf/path.h>

namespace omni::ui {
class DynamicTextureProvider;
}

namespace CesiumGltf {
struct ImageCesium;
struct MeshPrimitive;
struct Model;
} // namespace CesiumGltf

namespace cesium::omniverse {

class FabricTexture;

class FabricMaterial {
  public:
    FabricMaterial(
        pxr::SdfPath path,
        const FabricMaterialDefinition& materialDefinition,
        pxr::SdfAssetPath defaultTextureAssetPath,
        long stageId);
    ~FabricMaterial();

    void setMaterial(int64_t tilesetId, const MaterialInfo& materialInfo);

    void setBaseColorTexture(const std::shared_ptr<FabricTexture>& texture, const TextureInfo& textureInfo);

    void clearBaseColorTexture();

    void setActive(bool active);

    [[nodiscard]] omni::fabric::Path getPathFabric() const;
    [[nodiscard]] const FabricMaterialDefinition& getMaterialDefinition() const;

  private:
    void initialize(pxr::SdfPath path, const FabricMaterialDefinition& materialDefinition);
    void reset();
    void setTilesetId(int64_t tilesetId);
    void setMaterialValues(const MaterialInfo& materialInfo);
    void setBaseColorTextureValues(const pxr::SdfAssetPath& textureAssetPath, const TextureInfo& textureInfo);
    bool stageDestroyed();

    const FabricMaterialDefinition _materialDefinition;
    const pxr::SdfAssetPath _defaultTextureAssetPath;
    const long _stageId;

    omni::fabric::Path _materialPathFabric;
    omni::fabric::Path _shaderPathFabric;
    omni::fabric::Path _baseColorTexPathFabric;
};

} // namespace cesium::omniverse
