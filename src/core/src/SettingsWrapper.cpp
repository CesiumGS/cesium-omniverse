#include "cesium/omniverse/SettingsWrapper.h"

#include <carb/InterfaceUtils.h>
#include <carb/settings/ISettings.h>
#include <spdlog/fmt/bundled/format.h>

namespace cesium::omniverse::Settings {

namespace {
const uint64_t MAX_SESSIONS = 10;
const std::string_view SESSION_ION_SERVER_URL_BASE =
    "/persistent/exts/cesium.omniverse/sessions/session{}/ionServerUrl";
const std::string_view SESSION_USER_ACCESS_TOKEN_BASE =
    "/persistent/exts/cesium.omniverse/sessions/session{}/userAccessToken";

std::string getIonApiUrlSettingPath(const uint64_t index) {
    return fmt::format(SESSION_ION_SERVER_URL_BASE, index);
}

std::string getAccessTokenSettingPath(const uint64_t index) {
    return fmt::format(SESSION_USER_ACCESS_TOKEN_BASE, index);
}

} // namespace

std::vector<AccessToken> getAccessTokens() {
    const auto iSettings = carb::getCachedInterface<carb::settings::ISettings>();

    std::vector<AccessToken> accessTokens;
    accessTokens.reserve(MAX_SESSIONS);

    for (uint64_t i = 0; i < MAX_SESSIONS; ++i) {
        const auto ionApiUrlKey = getIonApiUrlSettingPath(i);
        const auto accessTokenKey = getAccessTokenSettingPath(i);

        const auto ionApiUrlValue = iSettings->getStringBuffer(ionApiUrlKey.c_str());
        const auto accessTokenValue = iSettings->getStringBuffer(accessTokenKey.c_str());

        if (ionApiUrlValue && accessTokenValue) {
            // In C++ 20 this can be emplace_back without the {}
            accessTokens.push_back({ionApiUrlValue, accessTokenValue});
        }
    }

    return accessTokens;
}

void setAccessToken(const AccessToken& accessToken) {
    const auto iSettings = carb::getCachedInterface<carb::settings::ISettings>();

    const auto oldAccessTokens = getAccessTokens();

    std::vector<AccessToken> newAccessTokens;
    newAccessTokens.reserve(oldAccessTokens.size() + 1); // Worst case we'll be growing by 1, so preempt that.

    for (const auto& oldAccessToken : oldAccessTokens) {
        if (oldAccessToken.ionApiUrl == accessToken.ionApiUrl) {
            continue;
        }

        newAccessTokens.push_back(oldAccessToken);
    }

    newAccessTokens.push_back(accessToken);

    clearTokens();

    for (uint64_t i = 0; i < newAccessTokens.size(); ++i) {
        const auto ionApiUrlKey = getIonApiUrlSettingPath(i);
        const auto accessTokenKey = getAccessTokenSettingPath(i);

        iSettings->set(ionApiUrlKey.c_str(), newAccessTokens[i].ionApiUrl.c_str());
        iSettings->set(accessTokenKey.c_str(), newAccessTokens[i].accessToken.c_str());
    }
}

void removeAccessToken(const std::string& ionApiUrl) {
    const auto iSettings = carb::getCachedInterface<carb::settings::ISettings>();

    const auto oldAccessTokens = getAccessTokens();

    std::vector<AccessToken> newAccessTokens;
    newAccessTokens.reserve(oldAccessTokens.size());

    for (auto& oldAccessToken : oldAccessTokens) {
        if (oldAccessToken.ionApiUrl == ionApiUrl) {
            continue;
        }

        newAccessTokens.push_back(oldAccessToken);
    }

    clearTokens();

    for (uint64_t i = 0; i < newAccessTokens.size(); ++i) {
        const auto ionApiUrlKey = getIonApiUrlSettingPath(i);
        const auto accessTokenKey = getAccessTokenSettingPath(i);

        iSettings->set(ionApiUrlKey.c_str(), newAccessTokens[i].ionApiUrl.c_str());
        iSettings->set(accessTokenKey.c_str(), newAccessTokens[i].accessToken.c_str());
    }
}

void clearTokens() {
    const auto iSettings = carb::getCachedInterface<carb::settings::ISettings>();

    for (uint64_t i = 0; i < MAX_SESSIONS; ++i) {
        const auto serverKey = getIonApiUrlSettingPath(i);
        const auto tokenKey = getAccessTokenSettingPath(i);

        iSettings->destroyItem(serverKey.c_str());
        iSettings->destroyItem(tokenKey.c_str());
    }
}

} // namespace cesium::omniverse::Settings
