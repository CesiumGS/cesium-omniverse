#include "cesium/omniverse/TaskProcessor.h"

#include <spdlog/spdlog.h>

namespace Cesium {
void TaskProcessor::startTask(std::function<void()> f) {
    _dispatcher.Run(f);
}
} // namespace Cesium
