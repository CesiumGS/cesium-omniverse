#include "cesium/omniverse/CesiumIonSessionManager.h"

#include "cesium/omniverse/CesiumIonSession.h"
#include "cesium/omniverse/Context.h"

namespace cesium::omniverse {

CesiumIonSessionManager::CesiumIonSessionManager(Context* pContext)
    : _pContext(pContext) {}

std::shared_ptr<CesiumIonSession> CesiumIonSessionManager::getOrCreateSession(
    const std::string& ionServerUrl,
    const std::string& ionServerApiUrl,
    int64_t applicationId) {
    const auto key = ionServerUrl + ionServerApiUrl + std::to_string(applicationId);
    const auto foundIter = _sessions.find(key);
    if (foundIter != _sessions.end()) {
        return foundIter->second;
    }

    auto pSession = std::make_shared<CesiumIonSession>(
        _pContext->getAsyncSystem(), _pContext->getAssetAccessor(), ionServerUrl, ionServerApiUrl, applicationId);

    _sessions.emplace(key, pSession);

    return pSession;
}

} // namespace cesium::omniverse
