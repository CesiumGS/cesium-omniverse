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

        _baseColorFactor = GltfUtil::getBaseColorFactor(material);
        _metallicFactor = GltfUtil::getMetallicFactor(material);
        _roughnessFactor = GltfUtil::getRoughnessFactor(material);
        _hasBaseColorTexture = GltfUtil::getBaseColorTextureIndex(model, material).has_value();
    }

    if (hasImagery) {
        _hasBaseColorTexture = true;
    }
}

bool FabricMaterialDefinition::hasBaseColorTexture() const {
    return _hasBaseColorTexture;
}

bool FabricMaterialDefinition::hasTexcoordTransform() const {
    return _hasTexcoordTransform;
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

bool FabricMaterialDefinition::operator==(const FabricMaterialDefinition& other) const {
    if (_hasBaseColorTexture != other._hasBaseColorTexture) {
        return false;
    }

    if (_hasTexcoordTransform != other._hasTexcoordTransform) {
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

    return true;
}

} // namespace cesium::omniverse
