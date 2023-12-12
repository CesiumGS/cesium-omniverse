#pragma once

#include <pxr/usd/sdf/path.h>

namespace cesium::omniverse {
class OmniData {
  public:
    OmniData(const pxr::SdfPath& path);

    [[nodiscard]] pxr::SdfPath getPath() const;
    [[nodiscard]] pxr::SdfPath getSelectedIonServer() const;
    [[nodiscard]] bool getDebugDisableMaterials() const;
    [[nodiscard]] bool getDebugDisableTextures() const;
    [[nodiscard]] bool getDebugDisableGeometryPool() const;
    [[nodiscard]] bool getDebugDisableMaterialPool() const;
    [[nodiscard]] bool getDebugDisableTexturePool() const;
    [[nodiscard]] uint64_t getDebugGeometryPoolInitialCapacity() const;
    [[nodiscard]] uint64_t getDebugMaterialPoolInitialCapacity() const;
    [[nodiscard]] uint64_t getDebugTexturePoolInitialCapacity() const;
    [[nodiscard]] bool getDebugRandomColors() const;
    [[nodiscard]] bool getDebugDisableGeoreferencing() const;

  private:
    pxr::SdfPath _path;
};
} // namespace cesium::omniverse
