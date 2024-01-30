#pragma once

#include "cesium/omniverse/DataType.h"

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

    // Make sure to update this function when adding new fields to the struct
    // In C++ 20 we can use the default equality comparison (= default)
    // clang-format off
    bool operator==(const FabricPropertyDescriptor& other) const {
        return storageType == other.storageType &&
               type == other.type &&
               propertyId == other.propertyId &&
               featureIdSetIndex == other.featureIdSetIndex;
    }
    // clang-format on
};

} // namespace cesium::omniverse
