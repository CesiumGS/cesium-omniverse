#pragma once

#include <pxr/usd/sdf/path.h>

namespace cesium::omniverse {

class Context;

class OmniData {
  public:
    OmniData(Context* pContext, const pxr::SdfPath& path);
    ~OmniData() = default;
    OmniData(const OmniData&) = delete;
    OmniData& operator=(const OmniData&) = delete;
    OmniData(OmniData&&) noexcept = default;
    OmniData& operator=(OmniData&&) noexcept = default;

    [[nodiscard]] const pxr::SdfPath& getPath() const;
    [[nodiscard]] pxr::SdfPath getSelectedIonServerPath() const;
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
    Context* _pContext;
    pxr::SdfPath _path;
};
} // namespace cesium::omniverse
