#pragma once

#include <string>

namespace cesium::omniverse::Settings {

std::string getAccessToken();
void setAccessToken(const std::string& accessToken);
std::string getDefaultAccessToken();
void setDefaultAccessToken(const std::string& accessToken);

} // namespace cesium::omniverse::Settings
