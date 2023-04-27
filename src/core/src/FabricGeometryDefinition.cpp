#include "cesium/omniverse/FabricGeometryDefinition.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/GltfUtil.h"

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <CesiumGltf/Model.h>

namespace cesium::omniverse {

FabricGeometryDefinition::FabricGeometryDefinition(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    bool smoothNormals,
    bool hasImagery,
    uint64_t imageryTexcoordSetIndex) {

    const auto hasMaterial = GltfUtil::hasMaterial(primitive);
    const auto hasPrimitiveSt = GltfUtil::hasTexcoords(model, primitive, 0);
    const auto hasImagerySt = GltfUtil::hasImageryTexcoords(model, primitive, imageryTexcoordSetIndex);
    const auto hasNormals = GltfUtil::hasNormals(model, primitive, smoothNormals);

    _hasMaterial = (hasMaterial || hasImagery) && !Context::instance().getDebugDisableMaterials();
    _hasTexcoords = hasPrimitiveSt || hasImagerySt;
    _hasNormals = hasNormals;
    _doubleSided = GltfUtil::getDoubleSided(model, primitive);
}

bool FabricGeometryDefinition::hasMaterial() const {
    return _hasMaterial;
}

bool FabricGeometryDefinition::hasTexcoords() const {
    return _hasTexcoords;
}

bool FabricGeometryDefinition::hasNormals() const {
    return _hasNormals;
}

bool FabricGeometryDefinition::getDoubleSided() const {
    return _doubleSided;
}

bool FabricGeometryDefinition::operator==(const FabricGeometryDefinition& other) const {
    if (_hasMaterial != other._hasMaterial) {
        return false;
    }

    if (_hasTexcoords != other._hasTexcoords) {
        return false;
    }

    if (_hasNormals != other._hasNormals) {
        return false;
    }

    if (_doubleSided != other._doubleSided) {
        return false;
    }

    return true;
}

} // namespace cesium::omniverse
