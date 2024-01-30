#pragma once

#include <CesiumAsync/ITaskProcessor.h>
#include <pxr/base/work/dispatcher.h>

namespace cesium::omniverse {

class TaskProcessor final : public CesiumAsync::ITaskProcessor {
  public:
    TaskProcessor() = default;
    ~TaskProcessor() override = default;
    TaskProcessor(const TaskProcessor&) = delete;
    TaskProcessor& operator=(const TaskProcessor&) = delete;
    TaskProcessor(TaskProcessor&&) noexcept = delete;
    TaskProcessor& operator=(TaskProcessor&&) noexcept = delete;

    void startTask(std::function<void()> f) override;

  private:
    // TODO: should we being using something in Carbonite instead?
    pxr::WorkDispatcher _dispatcher;
};

} // namespace cesium::omniverse
