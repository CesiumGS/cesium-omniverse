#pragma once

#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/sdf/path.h>

#include <array>

namespace cesium::omniverse {

// This is a helper class for reading and writing Fabric asset attributes.

// We don't actually know the struct layout for assets since they are internal to Fabric
// but we know that the first field is an asset path and the second field is a resolved path
// corresponding to SdfAssetPath::GetAssetPath and SdfAssetPath::GetResolvedPath and that
// the remaining 48 bytes are zero (at least from what we've seen so far)

class FabricAsset {
  public:
    FabricAsset(const pxr::SdfAssetPath& assetPath);

    bool isEmpty() const;
    bool isPaddingEmpty() const;
    const char* getAssetPath() const;
    const char* getResolvedPath() const;

  private:
    pxr::TfToken _assetPath;
    pxr::TfToken _resolvedPath;
    std::array<std::byte, 48> _padding{};
};

} // namespace cesium::omniverse
