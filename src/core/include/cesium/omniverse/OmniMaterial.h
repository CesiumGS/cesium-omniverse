#pragma once

#include "cesium/omniverse/OmniMaterialDefinition.h"

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

class OmniMaterial {
  public:
    OmniMaterial(pxr::SdfPath path, const OmniMaterialDefinition& materialDefinition);
    virtual ~OmniMaterial() = default;

    virtual void setTile(
        int64_t tilesetId,
        int64_t tileId,
        const CesiumGltf::Model& model,
        const CesiumGltf::MeshPrimitive& primitive,
        const CesiumGltf::ImageCesium* imagery) = 0;

    void setActive(bool active);

    pxr::SdfPath getPath() const;
    const OmniMaterialDefinition& getMaterialDefinition() const;

  protected:
    virtual void reset() = 0;

    const pxr::SdfPath _path;
    const OmniMaterialDefinition _materialDefinition;
};

} // namespace cesium::omniverse
