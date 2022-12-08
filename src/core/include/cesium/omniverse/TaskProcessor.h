#pragma once

#include <CesiumAsync/ITaskProcessor.h>
#include <pxr/base/work/dispatcher.h>

namespace cesium::omniverse {
class TaskProcessor : public CesiumAsync::ITaskProcessor {
  public:
    void startTask(std::function<void()> f) override;

  private:
    pxr::WorkDispatcher _dispatcher;
};
} // namespace cesium::omniverse
