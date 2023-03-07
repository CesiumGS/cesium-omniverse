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

    int64_t ionAssetId;
    rasterOverlay.GetIonAssetIdAttr().Get<int64_t>(&ionAssetId);

    return ionAssetId;
}

std::optional<CesiumIonClient::Token> OmniIonRasterOverlay::getIonAccessToken() const {
    auto rasterOverlay = UsdUtil::getCesiumRasterOverlay(_path);

    std::string ionAccessToken;
    rasterOverlay.GetIonAccessTokenAttr().Get<std::string>(&ionAccessToken);

    if (ionAccessToken.empty()) {
        return Context::instance().getDefaultToken();
    }

    CesiumIonClient::Token t;
    t.token = ionAccessToken;

    return t;
}
} // namespace cesium::omniverse
