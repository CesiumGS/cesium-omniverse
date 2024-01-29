#pragma once

#include <spdlog/logger.h>

#include <unordered_set>

namespace cesium::omniverse {

class Logger final : public spdlog::logger {
  public:
    Logger();
    void oneTimeWarning(const std::string& warning);

  private:
    std::unordered_set<std::string> _oneTimeWarnings;
};

} // namespace cesium::omniverse
