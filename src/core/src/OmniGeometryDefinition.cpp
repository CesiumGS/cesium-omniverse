#include "cesium/omniverse/OmniGeometryDefinition.h"

#include "cesium/omniverse/GltfUtil.h"

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <CesiumGltf/Model.h>

namespace cesium::omniverse {

OmniGeometryDefinition::OmniGeometryDefinition(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    bool smoothNormals,
    bool hasImagery,
    uint64_t imageryTexcoordSetIndex,
    bool disableMaterials) {

    const auto hasMaterial = GltfUtil::hasMaterial(primitive);
    const auto hasPrimitiveSt = GltfUtil::hasTexcoords(model, primitive, 0);
    const auto hasImagerySt = GltfUtil::hasImageryTexcoords(model, primitive, imageryTexcoordSetIndex);

    _hasMaterial = (hasMaterial || hasImagery) && !disableMaterials;
    _hasTexcoords = hasPrimitiveSt || hasImagerySt;
    _hasNormals = GltfUtil::hasNormals(model, primitive, smoothNormals);
    _hasVertexColors = GltfUtil::hasVertexColors(model, primitive, 0);
    _doubleSided = GltfUtil::getDoubleSided(model, primitive);
}

bool OmniGeometryDefinition::hasMaterial() const {
    return _hasMaterial;
}

bool OmniGeometryDefinition::hasTexcoords() const {
    return _hasTexcoords;
}

bool OmniGeometryDefinition::hasNormals() const {
    return _hasNormals;
}

bool OmniGeometryDefinition::hasVertexColors() const {
    return _hasVertexColors;
}

bool OmniGeometryDefinition::getDoubleSided() const {
    return _doubleSided;
}

bool OmniGeometryDefinition::operator==(const OmniGeometryDefinition& other) const {
    if (_hasMaterial != other._hasMaterial) {
        return false;
    }

    if (_hasTexcoords != other._hasTexcoords) {
        return false;
    }

    if (_hasNormals != other._hasNormals) {
        return false;
    }

    if (_hasVertexColors != other._hasVertexColors) {
        return false;
    }

    if (_doubleSided != other._doubleSided) {
        return false;
    }

    return true;
}

} // namespace cesium::omniverse
