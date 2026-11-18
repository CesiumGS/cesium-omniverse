#pragma once

#include "cesium/omniverse/CppUtil.h"

#include <spdlog/logger.h>

#include <unordered_set>

namespace cesium::omniverse {

class ILogger : public spdlog::logger {
  public:
    ILogger(std::string&& name, spdlog::sinks_init_list&& sinks);
    // NOLINTNEXTLINE(modernize-use-override)
    virtual ~ILogger() = 0;

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
