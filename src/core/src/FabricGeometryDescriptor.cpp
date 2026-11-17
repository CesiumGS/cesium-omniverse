#include "cesium/omniverse/FabricGeometryDescriptor.h"

#include "cesium/omniverse/FabricVertexAttributeDescriptor.h"
#include "cesium/omniverse/FeaturesInfo.h"
#include "cesium/omniverse/FeaturesUtil.h"
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
    const FeaturesInfo& featuresInfo,
    bool smoothNormals)
    : _hasNormals(GltfUtil::hasNormals(model, primitive, smoothNormals))
    , _hasVertexColors(GltfUtil::hasVertexColors(model, primitive, 0))
    , _hasVertexIds(FeaturesUtil::hasFeatureIdType(featuresInfo, FeatureIdType::INDEX))
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

} // namespace cesium::omniverse
