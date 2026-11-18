#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cesium::omniverse::KitSettings {

struct AccessToken {
    std::string ionApiUrl;
    std::string accessToken;
};

std::vector<AccessToken> getAccessTokens();
void setAccessToken(const AccessToken& accessToken);
void removeAccessToken(const std::string& ionApiUrl);
void clearTokens();

uint64_t getMaxCacheItems();

} // namespace cesium::omniverse::KitSettings
