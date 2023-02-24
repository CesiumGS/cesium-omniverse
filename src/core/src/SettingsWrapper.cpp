#include "cesium/omniverse/SettingsWrapper.h"

#include <carb/InterfaceUtils.h>
#include <carb/settings/ISettings.h>

namespace cesium::omniverse::Settings {

namespace {
const char* PERSISTENT_SETTINGS_PREFIX = "/persistent";
const char* USER_ACCESS_TOKEN_PATH = "/exts/cesium.omniverse/userAccessToken";
const char* DEFAULT_ACCESS_TOKEN_PATH = "/exts/cesium.omniverse/defaultAccessToken";
} // namespace

std::string getAccessToken() {
    auto settings = carb::getCachedInterface<carb::settings::ISettings>();
    auto key = std::string(PERSISTENT_SETTINGS_PREFIX).append(USER_ACCESS_TOKEN_PATH).c_str();
    return {settings->getStringBuffer(key)};
}

void setAccessToken(const std::string& accessToken) {
    auto settings = carb::getCachedInterface<carb::settings::ISettings>();
    auto key = std::string(PERSISTENT_SETTINGS_PREFIX).append(USER_ACCESS_TOKEN_PATH).c_str();
    settings->set(key, accessToken.c_str());
}

std::string getDefaultAccessToken() {
    auto settings = carb::getCachedInterface<carb::settings::ISettings>();
    return {settings->getStringBuffer(DEFAULT_ACCESS_TOKEN_PATH)};
}

void setDefaultAccessToken(const std::string& accessToken) {
    auto settings = carb::getCachedInterface<carb::settings::ISettings>();
    settings->set(DEFAULT_ACCESS_TOKEN_PATH, accessToken.c_str());
}

} // namespace cesium::omniverse::Settings
