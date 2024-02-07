#pragma once

#include <string>
#include <vector>

namespace cesium::omniverse::Settings {

struct AccessToken {
    std::string ionApiUrl;
    std::string accessToken;
};

std::vector<AccessToken> getAccessTokens();
void setAccessToken(const AccessToken& accessToken);
void removeAccessToken(const std::string& ionApiUrl);
void clearTokens();

} // namespace cesium::omniverse::Settings
