#pragma once

#ifdef CESIUM_OMNI_MSVC
#pragma warning(push)
#pragma warning(disable : 4244 4305 4100 4201 4127 4251)
#endif

#include <pxr/usd/sdf/path.h>

#ifdef CESIUM_OMNI_MSVC
#pragma warning(pop)
#endif
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
