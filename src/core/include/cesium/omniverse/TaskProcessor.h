#pragma once

#include <CesiumAsync/ITaskProcessor.h>

#ifdef CESIUM_OMNI_MSVC
#pragma warning(push)
#pragma warning(disable : 4244 4305 4100 4201 4127 4251)
#endif

#include <pxr/base/work/dispatcher.h>

#ifdef CESIUM_OMNI_MSVC
#pragma warning(pop)
#endif

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
