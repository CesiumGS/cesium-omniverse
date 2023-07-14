#pragma once

#include <glm/glm.hpp>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>

namespace CesiumGltf {
struct MeshPrimitive;
struct Model;
} // namespace CesiumGltf

namespace cesium::omniverse {

struct MaterialInfo;

class FabricMaterialDefinition {
  public:
    FabricMaterialDefinition(const MaterialInfo& materialInfo, bool hasImagery, bool disableTextures);

    [[nodiscard]] bool hasBaseColorTexture() const;
    [[nodiscard]] bool hasVertexColors() const;

    bool operator==(const FabricMaterialDefinition& other) const;

  private:
    bool _hasBaseColorTexture;
    bool _hasVertexColors;
};

} // namespace cesium::omniverse
