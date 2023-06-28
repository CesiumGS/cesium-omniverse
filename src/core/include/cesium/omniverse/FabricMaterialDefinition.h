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
        bool hasImagery,
        bool disableTextures);

    [[nodiscard]] bool hasBaseColorTexture() const;
    [[nodiscard]] bool hasVertexColors() const;

    [[nodiscard]] float getAlphaCutoff() const;
    [[nodiscard]] int getAlphaMode() const;
    [[nodiscard]] float getBaseAlpha() const;
    [[nodiscard]] pxr::GfVec3f getBaseColorFactor() const;
    [[nodiscard]] pxr::GfVec3f getEmissiveFactor() const;
    [[nodiscard]] float getMetallicFactor() const;
    [[nodiscard]] float getRoughnessFactor() const;
    [[nodiscard]] int getWrapS() const;
    [[nodiscard]] int getWrapT() const;

    bool operator==(const FabricMaterialDefinition& other) const;

  private:
    bool _hasBaseColorTexture;
    bool _hasVertexColors;

    // Remove these once dynamic material values are supported in Kit 105
    float _alphaCutoff;
    int _alphaMode;
    float _baseAlpha;
    pxr::GfVec3f _baseColorFactor;
    pxr::GfVec3f _emissiveFactor;
    float _metallicFactor;
    float _roughnessFactor;
    int _wrapS;
    int _wrapT;
};

} // namespace cesium::omniverse
