#pragma once

#include <pxr/usd/sdf/path.h>

namespace CesiumIonClient {
struct Token;
}

namespace cesium::omniverse {

class CesiumIonSession;
class Context;

class OmniIonServer {
  public:
    OmniIonServer(Context* pContext, const pxr::SdfPath& path);
    ~OmniIonServer() = default;
    OmniIonServer(const OmniIonServer&) = delete;
    OmniIonServer& operator=(const OmniIonServer&) = delete;
    OmniIonServer(OmniIonServer&&) noexcept = default;
    OmniIonServer& operator=(OmniIonServer&&) noexcept = default;

    [[nodiscard]] const pxr::SdfPath& getPath() const;
    [[nodiscard]] std::string getIonServerUrl() const;
    [[nodiscard]] std::string getIonServerApiUrl() const;
    [[nodiscard]] int64_t getIonServerApplicationId() const;
    [[nodiscard]] CesiumIonClient::Token getToken() const;

    void setToken(const CesiumIonClient::Token& token);

    [[nodiscard]] std::shared_ptr<CesiumIonSession> getSession() const;

  private:
    Context* _pContext;
    pxr::SdfPath _path;
    std::shared_ptr<CesiumIonSession> _session;
};
} // namespace cesium::omniverse
