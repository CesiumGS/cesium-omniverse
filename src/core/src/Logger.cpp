#include "cesium/omniverse/Logger.h"

#include "cesium/omniverse/CppUtil.h"
#include "cesium/omniverse/LoggerSink.h"

namespace cesium::omniverse {

Logger::Logger()
    : spdlog::logger(
          std::string("cesium-omniverse"),
          spdlog::sinks_init_list{
              std::make_shared<LoggerSink>(omni::log::Level::eVerbose),
              std::make_shared<LoggerSink>(omni::log::Level::eInfo),
              std::make_shared<LoggerSink>(omni::log::Level::eWarn),
              std::make_shared<LoggerSink>(omni::log::Level::eError),
              std::make_shared<LoggerSink>(omni::log::Level::eFatal),
          }) {}

void Logger::oneTimeWarning(const std::string& warning) {
    if (CppUtil::contains(_oneTimeWarnings, warning)) {
        return;
    }

    _oneTimeWarnings.insert(warning);
    warn(warning);
}

} // namespace cesium::omniverse
