#include "cesium/omniverse/SessionRegistry.h"

#include "cesium/omniverse/CesiumIonSession.h"

namespace cesium::omniverse {

void SessionRegistry::addSession(
    CesiumAsync::AsyncSystem& asyncSystem,
    const std::shared_ptr<HttpAssetAccessor>& httpAssetAccessor,
    const pxr::SdfPath& ionServerPath) {
    auto prim = UsdUtil::getOrCreateIonServer(ionServerPath);

    std::string serverUrl;
    prim.GetIonServerUrlAttr().Get(&serverUrl);

    std::string apiUrl;
    prim.GetIonServerApiUrlAttr().Get(&apiUrl);

    int64_t applicationId;
    prim.GetIonServerApplicationIdAttr().Get(&applicationId);

    auto session = std::make_shared<CesiumIonSession>(asyncSystem, httpAssetAccessor, serverUrl, apiUrl, applicationId);
    session->resume();

    _sessions.insert({ionServerPath, std::move(session)});
}

std::vector<std::shared_ptr<CesiumIonSession>> SessionRegistry::getAllSessions() {
    std::vector<std::shared_ptr<CesiumIonSession>> sessions;
    sessions.reserve(_sessions.size());

    for (const auto& item : _sessions) {
        sessions.emplace_back(item.second);
    }

    return sessions;
}

std::vector<pxr::SdfPath> SessionRegistry::getAllServerPaths() {
    std::vector<pxr::SdfPath> paths;
    paths.reserve(_sessions.size());

    for (const auto& item : _sessions) {
        paths.emplace_back(item.first);
    }

    return paths;
}

std::shared_ptr<CesiumIonSession> SessionRegistry::getSession(const pxr::SdfPath& ionServerPath) {
    return _sessions[ionServerPath];
}

void SessionRegistry::removeSession(const pxr::SdfPath& ionServerPath) {
    if (auto pair = _sessions.find(ionServerPath); pair != _sessions.end()) {
        auto session = pair->second;

        session->disconnect();

        _sessions.erase(ionServerPath);
    }
}

bool SessionRegistry::sessionExists(const pxr::SdfPath& ionServerPath) {
    return _sessions.count(ionServerPath) > 0;
}

void SessionRegistry::clear() {
    _sessions.clear();
}

} // namespace cesium::omniverse
