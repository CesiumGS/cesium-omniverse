#pragma once
#include <carb/Interface.h>

namespace cesium::omniverse::tests {

class ICesiumOmniverseCppTestsInterface {
  public:
    CARB_PLUGIN_INTERFACE("cesium::omniverse::tests::ICesiumOmniverseCppTestsInterface", 0, 0);
    /**
     * @brief Call this on extension startup.
     *
     * @param cesiumExtensionLocation Path to the Cesium Omniverse extension location.
     */
    virtual void onStartup(const char* cesiumExtensionLocation) noexcept = 0;

    /**
     * @brief Call this on extension shutdown.
     */
    virtual void onShutdown() noexcept = 0;

    /**
     * @brief To be run at least one fram prior to `runAllTests` in order to
     * allow time for USD notifications to propogate.
     */
    virtual void setUpTests(long int stage_id) noexcept = 0;

    /**
     * @brief Collects and runs all the doctest tests defined in adjacent .cpp files
     */
    virtual void runAllTests() noexcept = 0;
};

} // namespace cesium::omniverse::tests
