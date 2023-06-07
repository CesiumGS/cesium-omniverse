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

    float getAlphaCutoff() const;
    int getAlphaMode() const;
    float getBaseAlpha() const;
    pxr::GfVec3f getBaseColorFactor() const;
    pxr::GfVec3f getEmissiveFactor() const;
    float getMetallicFactor() const;
    float getRoughnessFactor() const;
    int getWrapS() const;
    int getWrapT() const;

    bool operator==(const FabricMaterialDefinition& other) const;

  private:
    bool _hasBaseColorTexture{false};

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
