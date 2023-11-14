#pragma once

namespace cesium::omniverse {

class Profiler {
  public:
    Profiler(const Profiler&) = delete;
    Profiler(Profiler&&) = delete;

    static Profiler& getInstance() {
        static Profiler instance;
        return instance;
    }

    void initializeProfiling(const char* fileIdentifier);
    void shutDownProfiling();

  private:
    Profiler() = default;
    ~Profiler() = default;
};

} // namespace cesium::omniverse
