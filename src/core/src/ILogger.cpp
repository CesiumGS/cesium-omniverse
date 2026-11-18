#include "cesium/omniverse/ILogger.h"

namespace cesium::omniverse {

ILogger::ILogger(std::string&& name, spdlog::sinks_init_list&& sinks)
    : spdlog::logger(std::move(name), sinks) {}

} // namespace cesium::omniverse
