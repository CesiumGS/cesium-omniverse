#include "cesium/omniverse/OmniIonServer.h"

#include "cesium/omniverse/UsdUtil.h"

#include <CesiumUsdSchemas/ionServer.h>

namespace cesium::omniverse {

OmniIonServer::OmniIonServer(const pxr::SdfPath& path)
    : _path(path) {}

pxr::SdfPath OmniIonServer::getPath() const {
    return _path;
}

std::string OmniIonServer::getIonServerUrl() const {
    const auto ionServer = UsdUtil::getCesiumIonServer(_path);

    std::string ionServerUrl;
    ionServer.GetIonServerUrlAttr().Get(&ionServerUrl);

    return ionServerUrl;
}

std::string OmniIonServer::getIonServerApiUrl() const {
    const auto ionServer = UsdUtil::getCesiumIonServer(_path);

    std::string ionServerApiUrl;
    ionServer.GetIonServerApiUrlAttr().Get(&ionServerApiUrl);

    return ionServerApiUrl;
}

int64_t OmniIonServer::getIonServerApplicationId() const {
    const auto ionServer = UsdUtil::getCesiumIonServer(_path);

    int64_t ionServerApplicationId;
    ionServer.GetIonServerApplicationIdAttr().Get(&ionServerApplicationId);

    return ionServerApplicationId;
}

CesiumIonClient::Token OmniIonServer::getToken() const {
    const auto ionServer = UsdUtil::getCesiumIonServer(_path);

    std::string projectDefaultIonAccessToken;
    std::string projectDefaultIonAccessTokenId;

    ionServer.GetProjectDefaultIonAccessTokenAttr().Get(&projectDefaultIonAccessToken);
    ionServer.GetProjectDefaultIonAccessTokenIdAttr().Get(&projectDefaultIonAccessTokenId);

    CesiumIonClient::Token t;
    t.id = projectDefaultIonAccessTokenId;
    t.token = projectDefaultIonAccessToken;

    return t;
}

void OmniIonServer::setToken(const CesiumIonClient::Token& token) {
    const auto ionServer = UsdUtil::getCesiumIonServer(_path);

    ionServer.GetProjectDefaultIonAccessTokenAttr().Set<std::string>(token.token);
    ionServer.GetProjectDefaultIonAccessTokenIdAttr().Set<std::string>(token.id);
}

} // namespace cesium::omniverse
