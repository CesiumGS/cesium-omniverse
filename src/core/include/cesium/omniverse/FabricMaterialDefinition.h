#pragma once

#include <glm/glm.hpp>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>

namespace CesiumGltf {
struct MeshPrimitive;
struct Model;
} // namespace CesiumGltf

namespace cesium::omniverse {

class FabricMaterialDefinition {
  public:
    FabricMaterialDefinition(
        const CesiumGltf::Model& model,
        const CesiumGltf::MeshPrimitive& primitive,
        bool hasImagery);

    bool hasBaseColorTexture() const;
    bool hasTexcoordTransform() const;

    pxr::GfVec3f getBaseColorFactor() const;
    float getMetallicFactor() const;
    float getRoughnessFactor() const;

    bool operator==(const FabricMaterialDefinition& other) const;

  private:
    bool _hasBaseColorTexture{false};
    bool _hasTexcoordTransform{false};

    // Remove these once dynamic material values are supported in Kit 105
    pxr::GfVec3f _baseColorFactor{1.0, 1.0, 1.0};
    float _metallicFactor{0.0f};
    float _roughnessFactor{1.0f};
};

} // namespace cesium::omniverse
