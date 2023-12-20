#include "cesium/omniverse/OmniIonImagery.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/UsdUtil.h"

#include <CesiumUsdSchemas/ionImagery.h>

namespace cesium::omniverse {

OmniIonImagery::OmniIonImagery(const pxr::SdfPath& path)
    : OmniImagery(path) {}

int64_t OmniIonImagery::getIonAssetId() const {
    auto imagery = UsdUtil::getCesiumIonImagery(_path);

    int64_t ionAssetId;
    imagery.GetIonAssetIdAttr().Get<int64_t>(&ionAssetId);

    return ionAssetId;
}

std::string OmniIonImagery::getIonApiUrl() const {
    const auto ionServerPath = getIonServerPath();

    if (ionServerPath.IsEmpty()) {
        return {};
    }

    auto ionServerPrim = UsdUtil::getOrCreateIonServer(ionServerPath);

    std::string ionApiUrl;
    ionServerPrim.GetIonServerApiUrlAttr().Get(&ionApiUrl);

    return ionApiUrl;
}

pxr::SdfPath OmniIonImagery::getIonServerPath() const {
    auto imagery = UsdUtil::getCesiumIonImagery(_path);

    pxr::SdfPathVector targets;
    imagery.GetIonServerBindingRel().GetForwardedTargets(&targets);

    if (targets.size() < 1) {
        return {};
    }

    return targets[0];
}


std::optional<CesiumIonClient::Token> OmniIonImagery::getIonAccessToken() const {
    const auto imagery = UsdUtil::getCesiumIonImagery(_path);

    std::string ionAccessToken;
    imagery.GetIonAccessTokenAttr().Get<std::string>(&ionAccessToken);

    if (!ionAccessToken.empty()) {
        CesiumIonClient::Token t;
        t.token = ionAccessToken;
        return t;
    }

    const auto ionServerPath = getIonServerPath();

    if (ionServerPath.IsEmpty()) {
        return std::nullopt;
    }

    const auto ionServer = UsdUtil::getOrCreateIonServer(ionServerPath);

    std::string projectDefaultToken;
    std::string projectDefaultTokenId;

    ionServer.GetProjectDefaultIonAccessTokenAttr().Get(&projectDefaultToken);
    ionServer.GetProjectDefaultIonAccessTokenIdAttr().Get(&projectDefaultTokenId);

    if (projectDefaultToken.empty()) {
        return std::nullopt;
    }

    return CesiumIonClient::Token{projectDefaultTokenId, "", projectDefaultToken};
}


} // namespace cesium::omniverse
