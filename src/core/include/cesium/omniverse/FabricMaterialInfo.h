#pragma once

#include "cesium/omniverse/FabricTextureInfo.h"

#include <glm/glm.hpp>

#include <optional>

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

namespace cesium::omniverse {

/**
* @brief Matches gltf_alpha_mode in gltf/pbr.mdl.
*/
enum class FabricAlphaMode : int {
    OPAQUE = 0,
    MASK = 1,
    BLEND = 2,
};

/**
* @brief Material values parsed from a glTF material passed to {@link FabricMaterial::setMaterial}.
*/
struct FabricMaterialInfo {
    double alphaCutoff;
    FabricAlphaMode alphaMode;
    double baseAlpha;
    glm::dvec3 baseColorFactor;
    glm::dvec3 emissiveFactor;
    double metallicFactor;
    double roughnessFactor;
    bool doubleSided;
    bool hasVertexColors;
    std::optional<FabricTextureInfo> baseColorTexture;

    // Make sure to update this function when adding new fields to the struct
    // In C++ 20 we can use the default equality comparison (= default)
    // clang-format off
    bool operator==(const FabricMaterialInfo& other) const {
        return alphaCutoff == other.alphaCutoff &&
               alphaMode == other.alphaMode &&
               baseAlpha == other.baseAlpha &&
               baseColorFactor == other.baseColorFactor &&
               emissiveFactor == other.emissiveFactor &&
               metallicFactor == other.metallicFactor &&
               roughnessFactor == other.roughnessFactor &&
               doubleSided == other.doubleSided &&
               hasVertexColors == other.hasVertexColors &&
               baseColorTexture == other.baseColorTexture;
    }
    // clang-format on
};

} // namespace cesium::omniverse
