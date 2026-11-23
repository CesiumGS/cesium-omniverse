#pragma once

#include "cesium/omniverse/DataType.h"

#include <omni/fabric/IToken.h>

namespace cesium::omniverse {

struct FabricVertexAttributeDescriptor {
    DataType type;
    omni::fabric::Token fabricAttributeName;
    omni::fabric::Token fabricInterpolationName;
    std::string gltfAttributeName;

    bool operator==(const FabricVertexAttributeDescriptor& other) const = default;

    // This is needed for std::set to be sorted
    bool operator<(const FabricVertexAttributeDescriptor& other) const {
        return fabricAttributeName < other.fabricAttributeName;
    }
};

} // namespace cesium::omniverse
