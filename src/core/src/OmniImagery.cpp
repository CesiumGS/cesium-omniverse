#include "cesium/omniverse/OmniImagery.h"

#include "cesium/omniverse/AssetRegistry.h"
#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/FabricImageryLayersInfo.h"
#include "cesium/omniverse/Logger.h"
#include "cesium/omniverse/OmniIonServer.h"
#include "cesium/omniverse/UsdUtil.h"

#include <CesiumIonClient/Token.h>
#include <CesiumUsdSchemas/imagery.h>

namespace cesium::omniverse {

OmniImagery::OmniImagery(Context* pContext, const pxr::SdfPath& path)
    : _pContext(pContext)
    , _path(path) {}

const pxr::SdfPath& OmniImagery::getPath() const {
    return _path;
}

bool OmniImagery::getShowCreditsOnScreen() const {
    const auto cesiumImagery = UsdUtil::getCesiumImagery(_pContext->getUsdStage(), _path);

    bool showCreditsOnScreen;
    cesiumImagery.GetShowCreditsOnScreenAttr().Get(&showCreditsOnScreen);

    return showCreditsOnScreen;
}

double OmniImagery::getAlpha() const {
    const auto cesiumImagery = UsdUtil::getCesiumImagery(_pContext->getUsdStage(), _path);

    float alpha;
    cesiumImagery.GetAlphaAttr().Get(&alpha);

    return static_cast<double>(alpha);
}

FabricOverlayRenderMethod OmniImagery::getOverlayRenderMethod() const {
    const auto cesiumImagery = UsdUtil::getCesiumImagery(_pContext->getUsdStage(), _path);

    pxr::TfToken overlayRenderMethod;
    cesiumImagery.GetOverlayRenderMethodAttr().Get(&overlayRenderMethod);

    if (overlayRenderMethod == pxr::CesiumTokens->overlay) {
        return FabricOverlayRenderMethod::OVERLAY;
    } else if (overlayRenderMethod == pxr::CesiumTokens->clip) {
        return FabricOverlayRenderMethod::CLIPPING;
    }

    _pContext->getLogger()->warn("Invalid overlay render method encountered {}.", overlayRenderMethod.GetText());
    return FabricOverlayRenderMethod::OVERLAY;
}

} // namespace cesium::omniverse
