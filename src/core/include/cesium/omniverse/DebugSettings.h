#pragma once

#include <cstdint>

namespace cesium::omniverse {

struct DebugSettings {
    bool disableMaterials;
    bool disableTextures;
    bool disableGeometryPool;
    bool disableMaterialPool;
    bool disableTexturePool;
    uint64_t geometryPoolInitialCapacity;
    uint64_t materialPoolInitialCapacity;
    uint64_t texturePoolInitialCapacity;
    bool randomColors;
};

} // namespace cesium::omniverse
