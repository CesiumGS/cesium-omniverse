#pragma once

#include "cesium/omniverse/FabricVertexAttributeDescriptor.h"

#include <set>

namespace CesiumGltf {
struct Model;
struct MeshPrimitive;
} // namespace CesiumGltf

namespace cesium::omniverse::FabricGltfUtil {

std::set<FabricVertexAttributeDescriptor>
getCustomVertexAttributes(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive);

}
