#pragma once

#include "cesium/omniverse/TextureInfo.h"

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
enum class AlphaMode : int {
    OPAQUE = 0,
    MASK = 1,
    BLEND = 2,
};

/**
* @brief Material values parsed from a glTF material.
*/
struct MaterialInfo {
    double alphaCutoff;
    AlphaMode alphaMode;
    double baseAlpha;
    glm::dvec3 baseColorFactor;
    glm::dvec3 emissiveFactor;
    double metallicFactor;
    double roughnessFactor;
    bool doubleSided;
    bool hasVertexColors;
    std::optional<TextureInfo> baseColorTexture;

    bool operator==(const MaterialInfo& other) const = default;
};

} // namespace cesium::omniverse
