#pragma once

#include <optional>
#include <string>
#include <vector>

namespace cesium::omniverse::Settings {

struct UserAccessToken {
    std::string ionUrl;
    std::string token;
};

std::string getIonServerSettingPath(const size_t index);
std::string getUserAccessTokenSettingPath(const size_t index);
const std::vector<UserAccessToken> getAccessTokens();
void setAccessTokens(const std::vector<UserAccessToken>& accessTokens);
void clearTokens();

} // namespace cesium::omniverse::Settings
