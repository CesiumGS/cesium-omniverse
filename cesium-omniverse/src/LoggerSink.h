#pragma once

#include <spdlog/details/null_mutex.h>
#include <spdlog/logger.h>
#include <spdlog/sinks/base_sink.h>

#include <mutex>
#include <string>

namespace Cesium {
class LoggerSink : public spdlog::sinks::base_sink<spdlog::details::null_mutex> {
  protected:
    void sink_it_(const spdlog::details::log_msg& msg) override;

    void flush_() override;

  private:
    std::string formatMessage(const spdlog::details::log_msg& msg);

    std::mutex _formatMutex;
};
} // namespace Cesium
