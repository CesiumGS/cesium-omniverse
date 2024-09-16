#include "cesium/omniverse/OmniIonServer.h"

#include "cesium/omniverse/CesiumIonSession.h"
#include "cesium/omniverse/CesiumIonSessionManager.h"
#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/UsdUtil.h"

#include <CesiumUsdSchemas/ionServer.h>

namespace cesium::omniverse {

OmniIonServer::OmniIonServer(Context* pContext, const pxr::SdfPath& path)
    : _pContext(pContext)
    , _path(path) {
    getSession()->resume();
}

const pxr::SdfPath& OmniIonServer::getPath() const {
    return _path;
}

std::shared_ptr<CesiumIonSession> OmniIonServer::getSession() const {
    return _pContext->getCesiumIonSessionManager().getOrCreateSession(
        getIonServerUrl(), getIonServerApiUrl(), getIonServerApplicationId());
}

std::string OmniIonServer::getIonServerUrl() const {
    const auto cesiumIonServer = UsdUtil::getCesiumIonServer(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumIonServer)) {
        return "";
    }

    std::string ionServerUrl;
    cesiumIonServer.GetIonServerUrlAttr().Get(&ionServerUrl);

    return ionServerUrl;
}

std::string OmniIonServer::getIonServerApiUrl() const {
    const auto cesiumIonServer = UsdUtil::getCesiumIonServer(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumIonServer)) {
        return "";
    }

    std::string ionServerApiUrl;
    cesiumIonServer.GetIonServerApiUrlAttr().Get(&ionServerApiUrl);

    return ionServerApiUrl;
}

int64_t OmniIonServer::getIonServerApplicationId() const {
    const auto cesiumIonServer = UsdUtil::getCesiumIonServer(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumIonServer)) {
        return 0;
    }

    int64_t ionServerApplicationId;
    cesiumIonServer.GetIonServerApplicationIdAttr().Get(&ionServerApplicationId);

    return ionServerApplicationId;
}

CesiumIonClient::Token OmniIonServer::getToken() const {
    const auto cesiumIonServer = UsdUtil::getCesiumIonServer(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumIonServer)) {
        return {};
    }

    std::string projectDefaultIonAccessToken;
    std::string projectDefaultIonAccessTokenId;

    cesiumIonServer.GetProjectDefaultIonAccessTokenAttr().Get(&projectDefaultIonAccessToken);
    cesiumIonServer.GetProjectDefaultIonAccessTokenIdAttr().Get(&projectDefaultIonAccessTokenId);

    CesiumIonClient::Token t;
    t.id = projectDefaultIonAccessTokenId;
    t.token = projectDefaultIonAccessToken;

    return t;
}

void OmniIonServer::setToken(const CesiumIonClient::Token& token) {
    const auto cesiumIonServer = UsdUtil::getCesiumIonServer(_pContext->getUsdStage(), _path);
    if (!UsdUtil::isSchemaValid(cesiumIonServer)) {
        return;
    }

    cesiumIonServer.GetProjectDefaultIonAccessTokenAttr().Set(token.token);
    cesiumIonServer.GetProjectDefaultIonAccessTokenIdAttr().Set(token.id);
}

} // namespace cesium::omniverse
