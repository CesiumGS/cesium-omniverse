#pragma once

#include "cesium/omniverse/CppUtil.h"

#include <spdlog/logger.h>

#include <unordered_set>

namespace cesium::omniverse {

class Logger final : public spdlog::logger {
  public:
    Logger();

    template <typename T> void oneTimWarning(const T& warning) {
        if (CppUtil::contains(_oneTimeWarnings, warning)) {
            return;
        }

        _oneTimeWarnings.insert(warning);
        warn(warning);
    }

    template <typename... Args> void oneTimeWarning(spdlog::format_string_t<Args...> fmt, Args&&... args) {
        const auto warning = fmt::format(fmt, std::forward<Args>(args)...);

        if (CppUtil::contains(_oneTimeWarnings, warning)) {
            return;
        }

        _oneTimeWarnings.insert(warning);
        warn(warning);
    }

  private:
    std::unordered_set<std::string> _oneTimeWarnings;
};

} // namespace cesium::omniverse
