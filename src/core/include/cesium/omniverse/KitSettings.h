#pragma once

#include "cesium/omniverse/ISettings.h"

namespace cesium::omniverse {

class KitSettings : public ISettings {
  public:
    [[nodiscard]] std::vector<AccessToken> getAccessTokens() const override;
    void setAccessToken(const AccessToken& accessToken) const override;
    void removeAccessToken(const std::string& ionApiUrl) const override;
    void clearTokens() const override;
    [[nodiscard]] uint64_t getMaxCacheItems() const override;
};

} // namespace cesium::omniverse
