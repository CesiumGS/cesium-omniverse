#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cesium::omniverse {

struct AccessToken {
    std::string ionApiUrl;
    std::string accessToken;
};

class ISettings {
  public:
    virtual ~ISettings() = default;

    [[nodiscard]] virtual std::vector<AccessToken> getAccessTokens() const = 0;
    virtual void setAccessToken(const AccessToken& accessToken) const = 0;
    virtual void removeAccessToken(const std::string& ionApiUrl) const = 0;
    virtual void clearTokens() const = 0;
    [[nodiscard]] virtual uint64_t getMaxCacheItems() const = 0;
};

} // namespace cesium::omniverse
