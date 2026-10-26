#pragma once

#include "cesium/omniverse/DataType.h"

#include <cstdint>
#include <string>

namespace cesium::omniverse {

enum class FabricPropertyStorageType {
    ATTRIBUTE,
    TEXTURE,
    TABLE,
};

struct FabricPropertyDescriptor {
    FabricPropertyStorageType storageType;
    MdlInternalPropertyType type;
    std::string propertyId;
    uint64_t featureIdSetIndex; // Only relevant for property tables

    bool operator==(const FabricPropertyDescriptor& other) const = default;
};

} // namespace cesium::omniverse
