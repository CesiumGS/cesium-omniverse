#include "cesium/omniverse/FabricGeometryDefinition.h"

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
    bool smoothNormals) {
    _hasNormals = GltfUtil::hasNormals(model, primitive, smoothNormals);
    _hasVertexColors = GltfUtil::hasVertexColors(model, primitive, 0);
    _texcoordSetCount = GltfUtil::getTexcoordSetIndexes(model, primitive).size() +
                        GltfUtil::getImageryTexcoordSetIndexes(model, primitive).size();
}

bool FabricGeometryDefinition::hasNormals() const {
    return _hasNormals;
}

bool FabricGeometryDefinition::hasVertexColors() const {
    return _hasVertexColors;
}

[[nodiscard]] uint64_t FabricGeometryDefinition::getTexcoordSetCount() const {
    return _texcoordSetCount;
}

bool FabricGeometryDefinition::operator==(const FabricGeometryDefinition& other) const {
    if (_hasNormals != other._hasNormals) {
        return false;
    }

    if (_hasVertexColors != other._hasVertexColors) {
        return false;
    }

    if (_texcoordSetCount != other._texcoordSetCount) {
        return false;
    }

    return true;
}

} // namespace cesium::omniverse
