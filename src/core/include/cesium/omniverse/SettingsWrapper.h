#pragma once

#include <carb/InterfaceUtils.h>
#include <carb/settings/ISettings.h>

const char* PERSISTENT_SETTINGS_PREFIX = "/persistent";
const char* USER_ACCESS_TOKEN_PATH = "/exts/cesium.omniverse/userAccessToken";
const char* DEFAULT_ACCESS_TOKEN_PATH = "/exts/cesium.omniverse/defaultAccessToken";

namespace cesium::omniverse::Settings {

[[maybe_unused]] static std::string getAccessToken() {
    auto settings = carb::getCachedInterface<carb::settings::ISettings>();
    auto key = std::string(PERSISTENT_SETTINGS_PREFIX).append(USER_ACCESS_TOKEN_PATH).c_str();
    return {settings->getStringBuffer(key)};
}

[[maybe_unused]] static void setAccessToken(const std::string& accessToken) {
    auto settings = carb::getCachedInterface<carb::settings::ISettings>();
    auto key = std::string(PERSISTENT_SETTINGS_PREFIX).append(USER_ACCESS_TOKEN_PATH).c_str();
    settings->set(key, accessToken.c_str());
}

[[maybe_unused]] static std::string getDefaultAccessToken() {
    auto settings = carb::getCachedInterface<carb::settings::ISettings>();
    return {settings->getStringBuffer(DEFAULT_ACCESS_TOKEN_PATH)};
}

[[maybe_unused]] static void setDefaultAccessToken(const std::string& accessToken) {
    auto settings = carb::getCachedInterface<carb::settings::ISettings>();
    settings->set(DEFAULT_ACCESS_TOKEN_PATH, accessToken.c_str());
}

} // namespace cesium::omniverse::Settings