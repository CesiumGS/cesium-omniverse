#pragma once

#include <filesystem>

namespace cesium::omniverse::FilesystemUtil {
std::filesystem::path getCesiumCacheDirectory();
std::filesystem::path getUserHomeDirectory();
} // namespace cesium::omniverse::FilesystemUtil
