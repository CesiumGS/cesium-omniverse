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
    const auto imagery = UsdUtil::getCesiumImagery(_path);

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

std::string OmniImagery::getIonApiUrl() const {
    const auto ionServerPath = getIonServerPath();

    if (ionServerPath.IsEmpty()) {
        return {};
    }

    auto ionServerPrim = UsdUtil::getOrCreateIonServer(ionServerPath);

    std::string ionApiUrl;
    ionServerPrim.GetIonServerApiUrlAttr().Get(&ionApiUrl);

    return ionApiUrl;
}

pxr::SdfPath OmniImagery::getIonServerPath() const {
    auto imagery = UsdUtil::getCesiumImagery(_path);

    pxr::SdfPathVector targets;
    imagery.GetIonServerBindingRel().GetForwardedTargets(&targets);

    if (targets.size() < 1) {
        return {};
    }

    return targets[0];
}

bool OmniImagery::getShowCreditsOnScreen() const {
    auto imagery = UsdUtil::getCesiumImagery(_path);

    bool showCreditsOnScreen;
    imagery.GetShowCreditsOnScreenAttr().Get<bool>(&showCreditsOnScreen);

    return showCreditsOnScreen;
}

double OmniImagery::getAlpha() const {
    auto imagery = UsdUtil::getCesiumImagery(_path);

    float alpha;
    imagery.GetAlphaAttr().Get<float>(&alpha);

    return static_cast<double>(alpha);
}

} // namespace cesium::omniverse
