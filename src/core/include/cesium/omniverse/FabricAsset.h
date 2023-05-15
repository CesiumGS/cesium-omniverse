#pragma once

#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/sdf/path.h>

#include <array>

namespace cesium::omniverse {

// This is a helper class for reading and writing Fabric asset attributes.

// We don't actually know the struct layout for assets since they are internal to Fabric
// but we know that the first field is an asset path and the second field is a resolved path
// corresponding to SdfAssetPath::GetAssetPath and SdfAssetPath::GetResolvedPath. On Windows
// the object is padded out to 64 bytes (zeroed). On Linux there is no padding.

class FabricAsset {
  public:
    FabricAsset(const pxr::SdfAssetPath& assetPath);

    bool isEmpty() const;
    const char* getAssetPath() const;
    const char* getResolvedPath() const;

  private:
    pxr::TfToken _assetPath;
    pxr::TfToken _resolvedPath;
};

static_assert(sizeof(FabricAsset) == 16);
} // namespace cesium::omniverse
