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

    virtual void runAllTests(long int stage_id) noexcept = 0;
};

} // namespace cesium::omniverse::tests
