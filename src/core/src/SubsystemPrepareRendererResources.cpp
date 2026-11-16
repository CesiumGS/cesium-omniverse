#include "cesium/omniverse/SubsystemPrepareRendererResources.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/OmniTileset.h"
#include "cesium/omniverse/UsdUtil.h"

namespace cesium::omniverse {

SubsystemPrepareRendererResources::SubsystemPrepareRendererResources(Context* pContext, OmniTileset* pTileset)
    : _pContext(pContext)
    , _pTileset(pTileset) {}

bool SubsystemPrepareRendererResources::tilesetExists() const {
    // When a tileset is deleted there's a short period between the prim being deleted and TfNotice notifying us about the change.
    // This function helps us know whether we should proceed with loading render resources.
    return _pTileset && _pContext->hasUsdStage() && UsdUtil::primExists(_pContext->getUsdStage(), _pTileset->getPath());
}

void SubsystemPrepareRendererResources::detachTileset() {
    _pTileset = nullptr;
}

} // namespace cesium::omniverse
