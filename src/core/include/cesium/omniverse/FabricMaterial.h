#pragma once

#include "cesium/omniverse/FabricMaterialDefinition.h"

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

class FabricMaterial {
  public:
    FabricMaterial(pxr::SdfPath path, const FabricMaterialDefinition& materialDefinition);
    ~FabricMaterial();

    void setTile(
        int64_t tilesetId,
        int64_t tileId,
        const CesiumGltf::Model& model,
        const CesiumGltf::MeshPrimitive& primitive,
        const CesiumGltf::ImageCesium* imagery);

    void setActive(bool active);

    [[nodiscard]] carb::flatcache::Path getPathFabric() const;
    [[nodiscard]] const FabricMaterialDefinition& getMaterialDefinition() const;

  private:
    void initialize(pxr::SdfPath path, const FabricMaterialDefinition& materialDefinition);
    void reset();
    void setInitialValues(const FabricMaterialDefinition& materialDefinition);

    const FabricMaterialDefinition _materialDefinition;

    carb::flatcache::Path _materialPathFabric;
    carb::flatcache::Path _shaderPathFabric;
    carb::flatcache::Path _displacementPathFabric;
    carb::flatcache::Path _surfacePathFabric;
    carb::flatcache::Path _baseColorTexPathFabric;

    std::unique_ptr<omni::ui::DynamicTextureProvider> _baseColorTexture;
};

} // namespace cesium::omniverse
