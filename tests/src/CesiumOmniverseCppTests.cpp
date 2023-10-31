#define CARB_EXPORTS
#define DOCTEST_CONFIG_IMPLEMENT
#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS

#include "CesiumOmniverseCppTests.h"

#include "UsdUtilTests.h"
#include "testUtils.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/LoggerSink.h"

#include <carb/PluginUtils.h>
#include <cesium/omniverse/UsdUtil.h>
#include <doctest/doctest.h>
#include <omni/fabric/IFabric.h>

#include <iostream>

namespace cesium::omniverse::tests {

class CesiumOmniverseCppTestsPlugin final : public ICesiumOmniverseCppTestsInterface {
  public:
    void onStartup(const char* cesiumExtensionLocation) noexcept override {
        Context::onStartup(cesiumExtensionLocation);
    }

    void onShutdown() noexcept override {
        Context::onShutdown();
    }

    void setUpTests(long int stage_id) noexcept override {
        // This runs after the stage has been created, but at least one frame
        // before runAllTests. This is to allow time for USD notifications to
        // propogate, as prims cannot be created and used on the same frame.

        CESIUM_LOG_INFO("Setting up Cesium Omniverse Tests with stage id: {}", stage_id);

        cesium::omniverse::UsdUtil::setUpUsdUtilTests(stage_id);
    }

    void runAllTests() noexcept override {
        CESIUM_LOG_INFO("Running Cesium Omniverse Tests");

        // construct a doctest context
        doctest::Context context;

        // Some tests contain relative paths rooted in the top level project dir
        // so we set this as the working directory
        std::filesystem::path oldWorkingDir = std::filesystem::current_path();
        std::filesystem::current_path(TEST_WORKING_DIRECTORY);

        // run test suites
        context.run();

        // restore the previous working directory
        std::filesystem::current_path(oldWorkingDir);

        CESIUM_LOG_INFO("Cesium Omniverse tests complete");

        CESIUM_LOG_INFO("Cleaning up after tests");
        cleanUpAfterTests();
        CESIUM_LOG_INFO("Cesium Omniverse test prims removed");
    }

    void cleanUpAfterTests() noexcept {
        // delete any test related prims here
        cesium::omniverse::UsdUtil::cleanUpUsdUtilTests();
    }
};

} // namespace cesium::omniverse::tests

const struct carb::PluginImplDesc pluginImplDesc = {
    "cesium.omniverse.cpp.tests.plugin",
    "Cesium Omniverse Tests Plugin.",
    "Cesium",
    carb::PluginHotReload::eDisabled,
    "dev"};

// NOLINTBEGIN
CARB_PLUGIN_IMPL(pluginImplDesc, cesium::omniverse::tests::CesiumOmniverseCppTestsPlugin)
// NOLINTEND

void fillInterface([[maybe_unused]] cesium::omniverse::tests::CesiumOmniverseCppTestsPlugin& iface) {}
