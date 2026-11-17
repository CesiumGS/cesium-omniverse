#pragma once

#include "cesium/omniverse/DataType.h"

#include <cstdint>
#include <string>

namespace cesium::omniverse {

enum class PropertyStorageType {
    ATTRIBUTE,
    TEXTURE,
    TABLE,
};

struct PropertyDescriptor {
    PropertyStorageType storageType;
    MdlInternalPropertyType type;
    std::string propertyId;
    uint64_t featureIdSetIndex; // Only relevant for property tables

    bool operator==(const PropertyDescriptor& other) const = default;
};

} // namespace cesium::omniverse
