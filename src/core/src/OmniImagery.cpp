#include "cesium/omniverse/OmniImagery.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/UsdUtil.h"

#include <CesiumUsdSchemas/imagery.h>

namespace cesium::omniverse {

OmniImagery::OmniImagery(const pxr::SdfPath& path)
    : _path(path) {}

pxr::SdfPath OmniImagery::getPath() const {
    return _path;
}

std::string OmniImagery::getName() const {
    auto imagery = UsdUtil::getCesiumImagery(_path);
    return imagery.GetPrim().GetName().GetString();
}

int64_t OmniImagery::getIonAssetId() const {
    auto imagery = UsdUtil::getCesiumImagery(_path);

    int64_t ionAssetId;
    imagery.GetIonAssetIdAttr().Get<int64_t>(&ionAssetId);

    return ionAssetId;
}

std::optional<CesiumIonClient::Token> OmniImagery::getIonAccessToken() const {
    auto imagery = UsdUtil::getCesiumImagery(_path);

    std::string ionAccessToken;
    imagery.GetIonAccessTokenAttr().Get<std::string>(&ionAccessToken);

    if (ionAccessToken.empty()) {
        return Context::instance().getDefaultToken();
    }

    CesiumIonClient::Token t;
    t.token = ionAccessToken;

    return t;
}
} // namespace cesium::omniverse
