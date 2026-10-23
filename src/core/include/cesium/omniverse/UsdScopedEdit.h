#ifdef CESIUM_OMNI_MSVC
#pragma warning(push)
#pragma warning(disable : 4244 4305 4100 4201 4127 4251)
#endif

#include <pxr/usd/usd/stage.h>

#ifdef CESIUM_OMNI_MSVC
#pragma warning(pop)
#endif

namespace cesium::omniverse {

class UsdScopedEdit {
  public:
    UsdScopedEdit(const pxr::UsdStageWeakPtr& pStage);
    ~UsdScopedEdit();
    UsdScopedEdit(const UsdScopedEdit&) = delete;
    UsdScopedEdit& operator=(const UsdScopedEdit&) = delete;
    UsdScopedEdit(UsdScopedEdit&&) noexcept = delete;
    UsdScopedEdit& operator=(UsdScopedEdit&&) noexcept = delete;

  private:
    pxr::UsdStageWeakPtr _pStage;
    pxr::SdfLayerHandle _sessionLayer;
    bool _sessionLayerWasEditable;
    pxr::UsdEditTarget _originalEditTarget;
};

} // namespace cesium::omniverse
