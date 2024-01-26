#include "cesium/omniverse/OmniIonServer.h"

#include "cesium/omniverse/CesiumIonSession.h"
#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/HttpAssetAccessor.h"
#include "cesium/omniverse/UsdUtil.h"

#include <CesiumUsdSchemas/ionServer.h>

namespace cesium::omniverse {

OmniIonServer::OmniIonServer(Context* pContext, const pxr::SdfPath& path)
    : _pContext(pContext)
    , _path(path)
    , _session(std::make_shared<CesiumIonSession>(
          pContext->getAsyncSystem(),
          pContext->getHttpAssetAccessor(),
          getIonServerUrl(),
          getIonServerApiUrl(),
          getIonServerApplicationId())) {}

const pxr::SdfPath& OmniIonServer::getPath() const {
    return _path;
}

std::shared_ptr<CesiumIonSession> OmniIonServer::getSession() const {
    return _session;
}

std::string OmniIonServer::getIonServerUrl() const {
    const auto cesiumIonServer = UsdUtil::getCesiumIonServer(_pContext->getUsdStage(), _path);

    std::string ionServerUrl;
    cesiumIonServer.GetIonServerUrlAttr().Get(&ionServerUrl);

    return ionServerUrl;
}

std::string OmniIonServer::getIonServerApiUrl() const {
    const auto cesiumIonServer = UsdUtil::getCesiumIonServer(_pContext->getUsdStage(), _path);

    std::string ionServerApiUrl;
    cesiumIonServer.GetIonServerApiUrlAttr().Get(&ionServerApiUrl);

    return ionServerApiUrl;
}

int64_t OmniIonServer::getIonServerApplicationId() const {
    const auto cesiumIonServer = UsdUtil::getCesiumIonServer(_pContext->getUsdStage(), _path);

    int64_t ionServerApplicationId;
    cesiumIonServer.GetIonServerApplicationIdAttr().Get(&ionServerApplicationId);

    return ionServerApplicationId;
}

CesiumIonClient::Token OmniIonServer::getToken() const {
    const auto cesiumIonServer = UsdUtil::getCesiumIonServer(_pContext->getUsdStage(), _path);

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

    cesiumIonServer.GetProjectDefaultIonAccessTokenAttr().Set(token.token);
    cesiumIonServer.GetProjectDefaultIonAccessTokenIdAttr().Set(token.id);
}

} // namespace cesium::omniverse
