#include "cesium/omniverse/OmniIonImagery.h"

#include <CesiumUsdSchemas/ionImagery.h>
#include "cesium/omniverse/UsdUtil.h"
#include "cesium/omniverse/Context.h"


namespace cesium::omniverse {

OmniIonImagery::OmniIonImagery(const pxr::SdfPath& path)
    : OmniImagery(path) {}

int64_t OmniIonImagery::getIonAssetId() const {
    auto imagery = UsdUtil::getCesiumImagery(_path);

    int64_t ionAssetId;
    imagery.GetIonAssetIdAttr().Get<int64_t>(&ionAssetId);

    return ionAssetId;
}

std::optional<CesiumIonClient::Token> OmniIonImagery::getIonAccessToken() const {
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
