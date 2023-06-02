#include "cesium/omniverse/FabricMaterialDefinition.h"

#include "cesium/omniverse/GltfUtil.h"

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <CesiumGltf/Model.h>

namespace cesium::omniverse {

FabricMaterialDefinition::FabricMaterialDefinition(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    bool hasImagery) {

    const auto hasGltfMaterial = GltfUtil::hasMaterial(primitive);

    if (hasGltfMaterial) {
        const auto& material = model.materials[static_cast<size_t>(primitive.material)];

        _alphaCutoff = GltfUtil::getAlphaCutoff(material);
        _alphaMode = GltfUtil::getAlphaMode(material);
        _baseAlpha = GltfUtil::getBaseAlpha(material);
        _baseColorFactor = GltfUtil::getBaseColorFactor(material);
        _metallicFactor = GltfUtil::getMetallicFactor(material);
        _roughnessFactor = GltfUtil::getRoughnessFactor(material);
        _hasBaseColorTexture = GltfUtil::getBaseColorTextureIndex(model, material).has_value();
        _wrapS = GltfUtil::getBaseColorTextureWrapS(model, material);
        _wrapT = GltfUtil::getBaseColorTextureWrapT(model, material);
    } else {
        _alphaCutoff = GltfUtil::getDefaultAlphaCutoff();
        _alphaMode = GltfUtil::getDefaultAlphaMode();
        _baseAlpha = GltfUtil::getDefaultBaseAlpha();
        _baseColorFactor = GltfUtil::getDefaultBaseColorFactor();
        _metallicFactor = 0.0f; // Override the glTF default of 1.0
        _roughnessFactor = GltfUtil::getDefaultRoughnessFactor();
        _hasBaseColorTexture = false;
        _wrapS = GltfUtil::getDefaultWrapS();
        _wrapT = GltfUtil::getDefaultWrapT();
    }

    if (hasImagery) {
        _hasBaseColorTexture = true;
        _wrapS = CesiumGltf::Sampler::WrapS::CLAMP_TO_EDGE;
        _wrapT = CesiumGltf::Sampler::WrapS::CLAMP_TO_EDGE;
    }
}

bool FabricMaterialDefinition::hasBaseColorTexture() const {
    return _hasBaseColorTexture;
}

float FabricMaterialDefinition::getAlphaCutoff() const {
    return _alphaCutoff;
}

int FabricMaterialDefinition::getAlphaMode() const {
    return _alphaMode;
}

float FabricMaterialDefinition::getBaseAlpha() const {
    return _baseAlpha;
}

pxr::GfVec3f FabricMaterialDefinition::getBaseColorFactor() const {
    return _baseColorFactor;
}

float FabricMaterialDefinition::getMetallicFactor() const {
    return _metallicFactor;
}

float FabricMaterialDefinition::getRoughnessFactor() const {
    return _roughnessFactor;
}

int FabricMaterialDefinition::getWrapS() const {
    return _wrapS;
}

int FabricMaterialDefinition::getWrapT() const {
    return _wrapT;
}

// In C++ 20 we can use the default equality comparison (= default)
bool FabricMaterialDefinition::operator==(const FabricMaterialDefinition& other) const {
    if (_hasBaseColorTexture != other._hasBaseColorTexture) {
        return false;
    }

    if (_alphaCutoff != other._alphaCutoff) {
        return false;
    }

    if (_alphaMode != other._alphaMode) {
        return false;
    }

    if (_baseAlpha != other._baseAlpha) {
        return false;
    }

    if (_baseColorFactor != other._baseColorFactor) {
        return false;
    }

    if (_metallicFactor != other._metallicFactor) {
        return false;
    }

    if (_roughnessFactor != other._roughnessFactor) {
        return false;
    }

    if (_wrapS != other._wrapS) {
        return false;
    }

    if (_wrapT != other._wrapT) {
        return false;
    }

    return true;
}

} // namespace cesium::omniverse
