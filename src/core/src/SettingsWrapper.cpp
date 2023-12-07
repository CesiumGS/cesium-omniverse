#include "cesium/omniverse/SettingsWrapper.h"

#include <carb/InterfaceUtils.h>
#include <carb/settings/ISettings.h>
#include <spdlog/fmt/bundled/format.h>

namespace cesium::omniverse::Settings {

namespace {
const size_t MAX_SESSIONS = 10;
const char* PERSISTENT_SETTINGS_PREFIX = "/persistent";
const char* SESSION_ION_SERVER_URL_BASE = "/exts/cesium.omniverse/sessions/session{}/ionServerUrl";
const char* SESSION_USER_ACCESS_TOKEN_BASE = "/exts/cesium.omniverse/sessions/session{}/userAccessToken";
} // namespace

std::string getIonServerSettingPath(const size_t index) {
    return std::string(PERSISTENT_SETTINGS_PREFIX).append(fmt::format(SESSION_ION_SERVER_URL_BASE, index));
}

std::string getUserAccessTokenSettingPath(const size_t index) {
    return std::string(PERSISTENT_SETTINGS_PREFIX).append(fmt::format(SESSION_USER_ACCESS_TOKEN_BASE, index));
}

const std::vector<UserAccessToken> getAccessTokens() {
    auto settings = carb::getCachedInterface<carb::settings::ISettings>();

    std::vector<UserAccessToken> tokens;
    tokens.reserve(MAX_SESSIONS);

    // I hate everything about this. -Adam
    for (size_t i = 0; i < MAX_SESSIONS; ++i) {
        const auto serverKey = getIonServerSettingPath(i);
        const auto ionUrlSetting = settings->getStringBuffer(serverKey.c_str());

        if (ionUrlSetting != nullptr) {
            const auto tokenKey = getUserAccessTokenSettingPath(i);
            const auto uatSetting = settings->getStringBuffer(tokenKey.c_str());

            UserAccessToken token;
            token.ionApiUrl = ionUrlSetting;
            token.token = uatSetting;

            tokens.emplace_back(token);
        }
    }

    return tokens;
}

void setAccessToken(const UserAccessToken& userAccessToken) {
    auto settings = carb::getCachedInterface<carb::settings::ISettings>();

    const auto accessTokens = getAccessTokens();

    std::vector<UserAccessToken> newAccessTokens;
    newAccessTokens.reserve(accessTokens.size() + 1); // Worst case we'll be growing by 1, so preempt that.

    for (const auto& accessToken : accessTokens) {
        if (accessToken.ionApiUrl == userAccessToken.ionApiUrl) {
            continue;
        }

        newAccessTokens.emplace_back(accessToken);
    }

    newAccessTokens.emplace_back(userAccessToken);

    clearTokens();

    for (size_t i = 0; i < newAccessTokens.size(); ++i) {
        const auto serverKey = getIonServerSettingPath(i);
        const auto tokenKey = getUserAccessTokenSettingPath(i);

        settings->set(serverKey.c_str(), newAccessTokens[i].ionApiUrl.c_str());
        settings->set(tokenKey.c_str(), newAccessTokens[i].token.c_str());
    }
}

void removeAccessToken(const std::string& ionApiUrl) {
    auto settings = carb::getCachedInterface<carb::settings::ISettings>();

    auto accessTokens = getAccessTokens();

    std::vector<UserAccessToken> newAccessTokens;
    newAccessTokens.reserve(accessTokens.size());

    for (auto& accessToken : accessTokens) {
        if (accessToken.ionApiUrl == ionApiUrl) {
            continue;
        }

        newAccessTokens.emplace_back(accessToken);
    }

    clearTokens();

    for (size_t i = 0; i < newAccessTokens.size(); ++i) {
        const auto serverKey = getIonServerSettingPath(i);
        const auto tokenKey = getUserAccessTokenSettingPath(i);

        settings->set(serverKey.c_str(), newAccessTokens[i].ionApiUrl.c_str());
        settings->set(tokenKey.c_str(), newAccessTokens[i].token.c_str());
    }
}

void clearTokens() {
    auto settings = carb::getCachedInterface<carb::settings::ISettings>();

    for (size_t i = 0; i < MAX_SESSIONS; ++i) {
        const auto serverKey = getIonServerSettingPath(i);
        const auto tokenKey = getUserAccessTokenSettingPath(i);

        settings->destroyItem(serverKey.c_str());
        settings->destroyItem(tokenKey.c_str());
    }
}

} // namespace cesium::omniverse::Settings
