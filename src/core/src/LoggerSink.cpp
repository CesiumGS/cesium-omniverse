#include "cesium/omniverse/LoggerSink.h"

#include <pxr/base/tf/callContext.h>
#include <pxr/base/tf/diagnostic.h>

using namespace pxr;

namespace cesium::omniverse {
void LoggerSink::sink_it_([[maybe_unused]] const spdlog::details::log_msg& msg) {
    TF_STATUS(formatMessage(msg));
}

void LoggerSink::flush_() {}

std::string LoggerSink::formatMessage(const spdlog::details::log_msg& msg) {
    // Frustratingly, spdlog::formatter isn't thread safe. So even though our sink
    // itself doesn't need to be protected by a mutex, the formatter does.
    // See https://github.com/gabime/spdlog/issues/897
    std::scoped_lock<std::mutex> lock(_formatMutex);

    spdlog::memory_buf_t formatted;
    formatter_->format(msg, formatted);
    return fmt::to_string(formatted);
}
} // namespace cesium::omniverse
