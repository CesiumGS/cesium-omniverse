#include "cesium/omniverse/FabricGeometryDescriptor.h"

#include "cesium/omniverse/FabricFeaturesInfo.h"
#include "cesium/omniverse/FabricFeaturesUtil.h"
#include "cesium/omniverse/FabricVertexAttributeDescriptor.h"
#include "cesium/omniverse/GltfUtil.h"

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <CesiumGltf/Model.h>

namespace cesium::omniverse {

FabricGeometryDescriptor::FabricGeometryDescriptor(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const FabricFeaturesInfo& featuresInfo,
    bool smoothNormals)
    : _hasNormals(GltfUtil::hasNormals(model, primitive, smoothNormals))
    , _hasVertexColors(GltfUtil::hasVertexColors(model, primitive, 0))
    , _hasVertexIds(FabricFeaturesUtil::hasFeatureIdType(featuresInfo, FabricFeatureIdType::INDEX))
    , _texcoordSetCount(
          GltfUtil::getTexcoordSetIndexes(model, primitive).size() +
          GltfUtil::getRasterOverlayTexcoordSetIndexes(model, primitive).size())
    , _customVertexAttributes(GltfUtil::getCustomVertexAttributes(model, primitive)) {}

bool FabricGeometryDescriptor::hasNormals() const {
    return _hasNormals;
}

bool FabricGeometryDescriptor::hasVertexColors() const {
    return _hasVertexColors;
}

bool FabricGeometryDescriptor::hasVertexIds() const {
    return _hasVertexIds;
}

uint64_t FabricGeometryDescriptor::getTexcoordSetCount() const {
    return _texcoordSetCount;
}

const std::set<FabricVertexAttributeDescriptor>& FabricGeometryDescriptor::getCustomVertexAttributes() const {
    return _customVertexAttributes;
}

bool FabricGeometryDescriptor::operator==(const FabricGeometryDescriptor& other) const {
    return _hasNormals == other._hasNormals && _hasVertexColors == other._hasVertexColors &&
           _hasVertexIds == other._hasVertexIds && _texcoordSetCount == other._texcoordSetCount &&
           _customVertexAttributes == other._customVertexAttributes;
}

} // namespace cesium::omniverse
