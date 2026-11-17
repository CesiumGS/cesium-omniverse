#pragma once

#include "cesium/omniverse/TextureInfo.h"

#include <optional>
#include <variant>
#include <vector>

namespace cesium::omniverse {

enum class FeatureIdType {
    INDEX,
    ATTRIBUTE,
    TEXTURE,
};

struct FeatureId {
    std::optional<uint64_t> nullFeatureId;
    uint64_t featureCount;
    std::variant<std::monostate, uint64_t, TextureInfo> featureIdStorage;
};

struct FeaturesInfo {
    std::vector<FeatureId> featureIds;
};

} // namespace cesium::omniverse
