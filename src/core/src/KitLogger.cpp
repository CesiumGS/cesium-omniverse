#include "cesium/omniverse/KitLogger.h"

#include "cesium/omniverse/KitLoggerSink.h"

namespace cesium::omniverse {

KitLogger::KitLogger()
    : ILogger(
          std::string("cesium-omniverse"),
          spdlog::sinks_init_list{
              std::make_shared<KitLoggerSink>(omni::log::Level::eVerbose),
              std::make_shared<KitLoggerSink>(omni::log::Level::eInfo),
              std::make_shared<KitLoggerSink>(omni::log::Level::eWarn),
              std::make_shared<KitLoggerSink>(omni::log::Level::eError),
              std::make_shared<KitLoggerSink>(omni::log::Level::eFatal),
          }) {}

} // namespace cesium::omniverse
