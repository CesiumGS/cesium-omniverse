#pragma once

#include <memory>
#include <string>
#include <unordered_map>

namespace cesium::omniverse {

class CesiumIonSession;
class Context;

class CesiumIonSessionManager {
  public:
    CesiumIonSessionManager(Context* pContext);
    ~CesiumIonSessionManager() = default;
    CesiumIonSessionManager(const CesiumIonSessionManager&) = delete;
    CesiumIonSessionManager& operator=(const CesiumIonSessionManager&) = delete;
    CesiumIonSessionManager(CesiumIonSessionManager&&) noexcept = delete;
    CesiumIonSessionManager& operator=(CesiumIonSessionManager&&) noexcept = delete;

    std::shared_ptr<CesiumIonSession>
    getOrCreateSession(const std::string& ionServerUrl, const std::string& ionServerApiUrl, int64_t applicationId);

  private:
    std::unordered_map<std::string, std::shared_ptr<CesiumIonSession>> _sessions;
    Context* _pContext;
};

} // namespace cesium::omniverse
