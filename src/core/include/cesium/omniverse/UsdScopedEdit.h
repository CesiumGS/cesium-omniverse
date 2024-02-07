#include <pxr/usd/usd/stage.h>

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
