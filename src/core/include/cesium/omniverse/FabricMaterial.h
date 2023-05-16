#pragma once

#include "cesium/omniverse/FabricMaterialDefinition.h"

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

    pxr::SdfPath getPath() const;
    const FabricMaterialDefinition& getMaterialDefinition() const;

  private:
    void initialize(pxr::SdfPath path, const FabricMaterialDefinition& materialDefinition);
    void reset();
    void setInitialValues(const FabricMaterialDefinition& materialDefinition);

    const FabricMaterialDefinition _materialDefinition;

    pxr::SdfPath _materialPath;
    pxr::SdfPath _shaderPath;

    std::unique_ptr<omni::ui::DynamicTextureProvider> _baseColorTexture;
};

} // namespace cesium::omniverse
