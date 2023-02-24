#include "cesium/omniverse/FabricAsset.h"

namespace cesium::omniverse {

FabricAsset::FabricAsset(const pxr::SdfAssetPath& assetPath)
    : _assetPath(assetPath.GetAssetPath())
    , _resolvedPath(assetPath.GetResolvedPath()) {}

bool FabricAsset::isEmpty() const {
    const std::byte* const bytes = reinterpret_cast<const std::byte*>(this);
    for (size_t i = 0; i < sizeof(FabricAsset); i++) {
        if (bytes[i] != std::byte(0)) {
            return false;
        }
    }
    return true;
}

bool FabricAsset::isPaddingEmpty() const {
#ifdef CESIUM_OMNI_WINDOWS
    const auto* const bytes = reinterpret_cast<const std::byte*>(&_padding);
    for (size_t i = 0; i < sizeof(_padding); i++) {
        if (bytes[i] != std::byte(0)) {
            return false;
        }
    }
#endif
    return true;
}

const char* FabricAsset::getAssetPath() const {
    assert(!isEmpty());
    return _assetPath.GetText();
}

const char* FabricAsset::getResolvedPath() const {
    assert(!isEmpty());
    return _resolvedPath.GetText();
}
}; // namespace cesium::omniverse
