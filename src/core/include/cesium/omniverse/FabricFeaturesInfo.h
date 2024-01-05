#pragma once

#include "cesium/omniverse/FabricTextureInfo.h"

#include <optional>
#include <variant>
#include <vector>

namespace cesium::omniverse {

enum class FabricFeatureIdType {
    INDEX,
    ATTRIBUTE,
    TEXTURE,
};

struct FabricFeatureId {
    std::optional<uint64_t> nullFeatureId;
    uint64_t featureCount;
    std::variant<std::monostate, uint64_t, FabricTextureInfo> featureIdStorage;
};

struct FabricFeaturesInfo {
    std::vector<FabricFeatureId> featureIds;
};

} // namespace cesium::omniverse
