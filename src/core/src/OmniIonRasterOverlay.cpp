#include "cesium/omniverse/OmniIonRasterOverlay.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/UsdUtil.h"

#include <CesiumUsdSchemas/rasterOverlay.h>

namespace cesium::omniverse {

OmniIonRasterOverlay::OmniIonRasterOverlay(const pxr::SdfPath& path)
    : _path(path) {}

pxr::SdfPath OmniIonRasterOverlay::getPath() const {
    return _path;
}

std::string OmniIonRasterOverlay::getName() const {
    auto rasterOverlay = UsdUtil::getCesiumRasterOverlay(_path);
    return rasterOverlay.GetPrim().GetName().GetString();
}

int64_t OmniIonRasterOverlay::getIonAssetId() const {
    auto rasterOverlay = UsdUtil::getCesiumRasterOverlay(_path);

    int64_t assetId;
    rasterOverlay.GetRasterOverlayIdAttr().Get<int64_t>(&assetId);

    return assetId;
}

std::optional<CesiumIonClient::Token> OmniIonRasterOverlay::getIonToken() const {
    auto rasterOverlay = UsdUtil::getCesiumRasterOverlay(_path);

    std::string ionToken;
    rasterOverlay.GetIonTokenAttr().Get<std::string>(&ionToken);

    if (ionToken.empty()) {
        return Context::instance().getDefaultToken();
    }

    CesiumIonClient::Token t;
    t.token = ionToken;

    return t;
}
} // namespace cesium::omniverse
