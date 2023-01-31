#pragma once

#include <carb/InterfaceUtils.h>
#include <carb/settings/ISettings.h>

const char* PERSISTENT_SETTINGS_PREFIX = "/persistent";
const char* ACCESS_TOKEN_PATH = "/exts/cesium.omniverse/accessToken";
const char* DEFAULT_ACCESS_TOKEN_PATH = "/exts/cesium.omniverse/defaultAccessToken";

namespace cesium::omniverse::Settings {

[[maybe_unused]] static std::string getAccessToken() {
    auto settings = carb::getCachedInterface<carb::settings::ISettings>();
    auto key = std::string(PERSISTENT_SETTINGS_PREFIX).append(ACCESS_TOKEN_PATH).c_str();
    return {settings->getStringBuffer(key)};
}

[[maybe_unused]] static void setAccessToken(const std::string& accessToken) {
    auto settings = carb::getCachedInterface<carb::settings::ISettings>();
    auto key = std::string(PERSISTENT_SETTINGS_PREFIX).append(ACCESS_TOKEN_PATH).c_str();
    settings->set(key, accessToken.c_str());
}

[[maybe_unused]] static std::string getDefaultAccessToken() {
    auto settings = carb::getCachedInterface<carb::settings::ISettings>();
    auto key = std::string(PERSISTENT_SETTINGS_PREFIX).append(DEFAULT_ACCESS_TOKEN_PATH).c_str();
    return {settings->getStringBuffer(key)};
}

} // namespace cesium::omniverse::Settings