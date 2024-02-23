#include "cesium/omniverse/FilesystemUtil.h"

#include <carb/Framework.h>
#include <carb/tokens/ITokens.h>
#include <carb/tokens/TokensUtils.h>

#include <cstdlib>
#include <filesystem>

#if defined(__linux__)
#include <pwd.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#endif

namespace cesium::omniverse::FilesystemUtil {

std::filesystem::path getCesiumCacheDirectory() {
    auto f = carb::getFramework();
    auto* tokensInterface = f->tryAcquireInterface<carb::tokens::ITokens>();
    std::string cacheDir;
    if (tokensInterface) {
        cacheDir = carb::tokens::resolveString(tokensInterface, "${omni_global_cache}");
    }
    if (!cacheDir.empty()) {
        std::filesystem::path cacheDirPath(cacheDir);
        if (exists(cacheDirPath)) {
            return cacheDirPath;
        }
        // Should we create the directory if it doesn't exist? It's hard to believe that Omniverse
        // won't have already created it.
    }
    std::string homeDir = getUserHomeDirectory();
    if (!homeDir.empty()) {
        std::filesystem::path homeDirPath(homeDir);
        auto cacheDirPath = homeDirPath / ".nvidia-omniverse";
        if (exists(cacheDirPath)) {
            return cacheDirPath;
        }
    }
    return {};
}

// Quite a lot of ceremony to get the home directory.

std::filesystem::path getUserHomeDirectory() {
    std::string homeDir;
#if defined(__linux__)
    if (char* cString = std::getenv("HOME")) {
        homeDir = cString;
    } else {
        passwd pwd;
        long bufsize = sysconf(_SC_GETPW_R_SIZE_MAX);
        if (bufsize == -1) {
            bufsize = 16384;
        }
        char* buf = new char[static_cast<size_t>(bufsize)];
        passwd* result = nullptr;
        int getResult = getpwuid_r(getuid(), &pwd, buf, static_cast<size_t>(bufsize), &result);
        if (getResult == 0) {
            homeDir = pwd.pw_dir;
        }
        delete[] buf;
    }
#elif defined(_WIN32)
    if (char* cString = std::getenv("USERPROFILE")) {
        homeDir = cString;
    }
#endif
    return homeDir;
}
} // namespace cesium::omniverse::FilesystemUtil
