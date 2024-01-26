#pragma once

#include <cstdint>

namespace cesium::omniverse {

struct AssetTroubleshootingDetails {
    int64_t assetId;
    bool assetExistsInUserAccount{false};
};

} // namespace cesium::omniverse
