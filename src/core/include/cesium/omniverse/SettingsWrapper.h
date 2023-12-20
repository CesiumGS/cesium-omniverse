#pragma once

#include <optional>
#include <string>
#include <vector>

namespace cesium::omniverse::Settings {

struct UserAccessToken {
    std::string ionApiUrl;
    std::string token;
};

std::string getIonServerSettingPath(const size_t index);
std::string getUserAccessTokenSettingPath(const size_t index);
const std::vector<UserAccessToken> getAccessTokens();
void setAccessToken(const UserAccessToken& userAccessToken);
void removeAccessToken(const std::string& ionApiUrl);
void clearTokens();

} // namespace cesium::omniverse::Settings
