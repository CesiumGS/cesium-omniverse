#pragma once

#include "cesium/omniverse/OmniMaterial.h"
#include "cesium/omniverse/OmniMaterialDefinition.h"

#include <carb/flatcache/IPath.h>
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

class FabricMaterial final : public OmniMaterial {
  public:
    FabricMaterial(pxr::SdfPath path, const OmniMaterialDefinition& materialDefinition);
    ~FabricMaterial();

    void setTile(
        int64_t tilesetId,
        int64_t tileId,
        const CesiumGltf::Model& model,
        const CesiumGltf::MeshPrimitive& primitive,
        const CesiumGltf::ImageCesium* imagery) override;

  private:
    void initialize(pxr::SdfPath path, const OmniMaterialDefinition& materialDefinition);
    void reset() override;
    void setInitialValues(const OmniMaterialDefinition& materialDefinition);

    carb::flatcache::Path _materialPathFabric;
    carb::flatcache::Path _shaderPathFabric;
    carb::flatcache::Path _displacementPathFabric;
    carb::flatcache::Path _surfacePathFabric;
    carb::flatcache::Path _baseColorTexPathFabric;

    std::unique_ptr<omni::ui::DynamicTextureProvider> _baseColorTexture;
};

} // namespace cesium::omniverse
