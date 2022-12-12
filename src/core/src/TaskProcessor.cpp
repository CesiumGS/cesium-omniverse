#include "cesium/omniverse/TaskProcessor.h"

#include <spdlog/spdlog.h>

namespace cesium::omniverse {
void TaskProcessor::startTask(std::function<void()> f) {
    _dispatcher.Run(f);
}
} // namespace cesium::omniverse
