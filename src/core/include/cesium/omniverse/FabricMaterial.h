#pragma once

#include "cesium/omniverse/FabricMaterialDefinition.h"

#include <omni/fabric/IPath.h>
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
        const CesiumGltf::ImageCesium* imagery,
        const glm::dvec2& imageryTexcoordTranslation,
        const glm::dvec2& imageryTexcoordScale);

    void setActive(bool active);

    [[nodiscard]] omni::fabric::Path getPathFabric() const;
    [[nodiscard]] const FabricMaterialDefinition& getMaterialDefinition() const;

  private:
    void initialize(pxr::SdfPath path, const FabricMaterialDefinition& materialDefinition);
    void reset();
    void setInitialValues(const FabricMaterialDefinition& materialDefinition);

    const FabricMaterialDefinition _materialDefinition;

    omni::fabric::Path _materialPathFabric;
    omni::fabric::Path _shaderPathFabric;
    omni::fabric::Path _baseColorTexPathFabric;

    std::unique_ptr<omni::ui::DynamicTextureProvider> _baseColorTexture;
};

} // namespace cesium::omniverse
