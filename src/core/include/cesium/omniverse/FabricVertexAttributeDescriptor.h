#pragma once

#include "cesium/omniverse/DataType.h"

#include <omni/fabric/IToken.h>

namespace cesium::omniverse {

struct FabricVertexAttributeDescriptor {
    DataType type;
    omni::fabric::Token fabricAttributeName;
    std::string gltfAttributeName;

    // Make sure to update this function when adding new fields to the struct
    // In C++ 20 we can use the default equality comparison (= default)
    // clang-format off
    bool operator==(const FabricVertexAttributeDescriptor& other) const {
        return type == other.type &&
               fabricAttributeName == other.fabricAttributeName &&
               gltfAttributeName == other.gltfAttributeName;
    }
    // clang-format on

    // This is needed for std::set to be sorted
    bool operator<(const FabricVertexAttributeDescriptor& other) const {
        return fabricAttributeName < other.fabricAttributeName;
    }
};

} // namespace cesium::omniverse
