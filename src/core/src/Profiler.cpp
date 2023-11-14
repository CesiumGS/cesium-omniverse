#include "cesium/omniverse/Profiler.h"

#include "cesium/omniverse/Context.h"

#include <CesiumUtility/Tracing.h>
#include <spdlog/fmt/fmt.h>
#include <windows.h>

#include <chrono>
#include <iostream>

namespace cesium::omniverse {

void Profiler::initializeProfiling(const char* fileIdentifier) {
    const auto timeNow = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::steady_clock::now());
    const auto timeSinceEpoch = timeNow.time_since_epoch().count();
    auto cesiumExtensionLocation = Context::instance().getCesiumExtensionLocation();
    const auto profileFilePath =
        cesiumExtensionLocation / fmt::format("cesium-trace-{}-{}.json", fileIdentifier, timeSinceEpoch);
    CESIUM_TRACE_INIT(profileFilePath.string());
    std::cout << profileFilePath << std::endl;
}
void Profiler::shutDownProfiling() {
    CESIUM_TRACE_SHUTDOWN();
}
} // namespace cesium::omniverse
