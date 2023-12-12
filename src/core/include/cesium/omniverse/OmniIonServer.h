#pragma once

#include <CesiumIonClient/Token.h>
#include <pxr/usd/sdf/path.h>

namespace cesium::omniverse {
class OmniIonServer {
  public:
    OmniIonServer(const pxr::SdfPath& path);

    [[nodiscard]] pxr::SdfPath getPath() const;
    [[nodiscard]] std::string getIonServerUrl() const;
    [[nodiscard]] std::string getIonServerApiUrl() const;
    [[nodiscard]] int64_t getIonServerApplicationId() const;
    [[nodiscard]] CesiumIonClient::Token getToken() const;

    void setToken(const CesiumIonClient::Token& token);

  private:
    pxr::SdfPath _path;
};
} // namespace cesium::omniverse
