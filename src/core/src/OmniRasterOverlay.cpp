#include "cesium/omniverse/OmniRasterOverlay.h"

#include "cesium/omniverse/AssetRegistry.h"
#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/FabricRasterOverlaysInfo.h"
#include "cesium/omniverse/Logger.h"
#include "cesium/omniverse/OmniIonServer.h"
#include "cesium/omniverse/UsdUtil.h"

#include <CesiumIonClient/Token.h>
#include <CesiumUsdSchemas/rasterOverlay.h>

namespace cesium::omniverse {

OmniRasterOverlay::OmniRasterOverlay(Context* pContext, const pxr::SdfPath& path)
    : _pContext(pContext)
    , _path(path) {}

const pxr::SdfPath& OmniRasterOverlay::getPath() const {
    return _path;
}

bool OmniRasterOverlay::getShowCreditsOnScreen() const {
    const auto cesiumRasterOverlay = UsdUtil::getCesiumRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumRasterOverlay)) {
        return false;
    }

    bool showCreditsOnScreen;
    cesiumRasterOverlay.GetShowCreditsOnScreenAttr().Get(&showCreditsOnScreen);

    return showCreditsOnScreen;
}

double OmniRasterOverlay::getAlpha() const {
    const auto cesiumRasterOverlay = UsdUtil::getCesiumRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumRasterOverlay)) {
        return 1.0;
    }

    float alpha;
    cesiumRasterOverlay.GetAlphaAttr().Get(&alpha);

    return static_cast<double>(alpha);
}

FabricOverlayRenderMethod OmniRasterOverlay::getOverlayRenderMethod() const {
    const auto cesiumRasterOverlay = UsdUtil::getCesiumRasterOverlay(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumRasterOverlay)) {
        return FabricOverlayRenderMethod::OVERLAY;
    }

    pxr::TfToken overlayRenderMethod;
    cesiumRasterOverlay.GetOverlayRenderMethodAttr().Get(&overlayRenderMethod);

    if (overlayRenderMethod == pxr::CesiumTokens->overlay) {
        return FabricOverlayRenderMethod::OVERLAY;
    } else if (overlayRenderMethod == pxr::CesiumTokens->clip) {
        return FabricOverlayRenderMethod::CLIPPING;
    }

    _pContext->getLogger()->warn("Invalid overlay render method encountered {}.", overlayRenderMethod.GetText());
    return FabricOverlayRenderMethod::OVERLAY;
}

} // namespace cesium::omniverse
