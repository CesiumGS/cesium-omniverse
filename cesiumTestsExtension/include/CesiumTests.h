#pragma once
#include <carb/Interface.h>
// TODO put this in the right place and try removing test sub-namespace
namespace cesium::omniverse::tests
{

class ICesiumOmniverseTestsInterface{
    public:
    CARB_PLUGIN_INTERFACE("cesium::omniverse::tests::ICesiumOmniverseTestsInterface", 0, 0);
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

    virtual void run_all_tests(long int stage_id) noexcept = 0;
};

}
