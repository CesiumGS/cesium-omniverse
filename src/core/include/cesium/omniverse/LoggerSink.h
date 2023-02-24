#pragma once

#include "cesium/omniverse/Context.h"

#include <omni/log/ILog.h>
#include <spdlog/details/null_mutex.h>
#include <spdlog/logger.h>
#include <spdlog/sinks/base_sink.h>

#include <mutex>
#include <string>

namespace cesium::omniverse {

#define CESIUM_LOG_VERBOSE(...) Context::instance().getLogger()->verbose(__VA_ARGS__)
#define CESIUM_LOG_INFO(...) Context::instance().getLogger()->info(__VA_ARGS__)
#define CESIUM_LOG_WARN(...) Context::instance().getLogger()->warn(__VA_ARGS__)
#define CESIUM_LOG_ERROR(...) Context::instance().getLogger()->error(__VA_ARGS__)
#define CESIUM_LOG_FATAL(...) Context::instance().getLogger()->fatal(__VA_ARGS__)

class LoggerSink : public spdlog::sinks::base_sink<spdlog::details::null_mutex> {
  public:
    LoggerSink(omni::log::Level logLevel);

  protected:
    void sink_it_(const spdlog::details::log_msg& msg) override;

    void flush_() override;

  private:
    std::string formatMessage(const spdlog::details::log_msg& msg);

    std::mutex _formatMutex;
    omni::log::Level _logLevel;
};

} // namespace cesium::omniverse
