#include "cesium/omniverse/FilesystemUtil.h"

#include <carb/Framework.h>
#include <carb/tokens/ITokens.h>
#include <carb/tokens/TokensUtils.h>

namespace cesium::omniverse {

std::filesystem::path getCesiumCacheDirectoryKit() {
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
    auto homeDirPath = FilesystemUtil::getUserHomeDirectory();
    if (!homeDirPath.empty()) {
        auto cacheDirPath = homeDirPath / ".nvidia-omniverse";
        if (exists(cacheDirPath)) {
            return cacheDirPath;
        }
    }
    return {};
}

} // namespace cesium::omniverse
