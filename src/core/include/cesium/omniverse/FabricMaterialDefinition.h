#pragma once

#include <glm/glm.hpp>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>

namespace cesium::omniverse {

struct MaterialInfo;

class FabricMaterialDefinition {
  public:
    FabricMaterialDefinition(const MaterialInfo& materialInfo, uint64_t imageryLayerCount, bool disableTextures);

    [[nodiscard]] bool hasVertexColors() const;
    [[nodiscard]] uint64_t getBaseColorTextureCount() const;
    [[nodiscard]] bool hasBaseColorTextures() const;

    // Make sure to update this function when adding new fields to the class
    bool operator==(const FabricMaterialDefinition& other) const;

  private:
    bool _hasVertexColors;
    uint64_t _baseColorTextureCount;
};

} // namespace cesium::omniverse
