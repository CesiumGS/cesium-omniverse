#include "cesium/omniverse/UsdScopedEdit.h"

namespace cesium::omniverse {

UsdScopedEdit::UsdScopedEdit(const pxr::UsdStageWeakPtr& pStage)
    : _pStage(pStage)
    , _sessionLayer(_pStage->GetSessionLayer())
    , _sessionLayerWasEditable(_sessionLayer->PermissionToEdit())
    , _originalEditTarget(_pStage->GetEditTarget()) {

    _sessionLayer->SetPermissionToEdit(true);
    _pStage->SetEditTarget(pxr::UsdEditTarget(_sessionLayer));
}

UsdScopedEdit::~UsdScopedEdit() {
    _sessionLayer->SetPermissionToEdit(_sessionLayerWasEditable);
    _pStage->SetEditTarget(_originalEditTarget);
}

} // namespace cesium::omniverse
