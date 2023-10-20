#pragma once

#include <glm/glm.hpp>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/usd/sdf/path.h>

namespace cesium::omniverse {

struct MaterialInfo;

class FabricMaterialDefinition {
  public:
    FabricMaterialDefinition(
        const MaterialInfo& materialInfo,
        uint64_t imageryLayerCount,
        bool disableTextures,
        const pxr::SdfPath& tilesetMaterialPath);

    [[nodiscard]] bool hasVertexColors() const;
    [[nodiscard]] bool hasBaseColorTexture() const;
    [[nodiscard]] uint64_t getImageryLayerCount() const;
    [[nodiscard]] bool hasTilesetMaterial() const;
    [[nodiscard]] const pxr::SdfPath& getTilesetMaterialPath() const;

    // Make sure to update this function when adding new fields to the class
    bool operator==(const FabricMaterialDefinition& other) const;

  private:
    bool _hasVertexColors;
    bool _hasBaseColorTexture;
    uint64_t _imageryLayerCount;
    pxr::SdfPath _tilesetMaterialPath;
};

} // namespace cesium::omniverse
