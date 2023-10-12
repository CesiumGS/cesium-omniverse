#define CARB_EXPORTS
#define DOCTEST_CONFIG_IMPLEMENT
#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS

#include "CesiumOmniverseCppTests.h"

#include "testUtils.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/LoggerSink.h"

#include <carb/PluginUtils.h>
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

    void runAllTests(long int stage_id) noexcept override {

        return;

        CESIUM_LOG_INFO("Running Cesium Omniverse Tests with stage id: {}", stage_id);

        Context::instance().setStageId(stage_id);

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

        CESIUM_LOG_INFO("Cesium Omniverse Tests complete");
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
// CARB_PLUGIN_IMPL_DEPS(omni::fabric::IFabric, omni::fabric::IStageReaderWriter)
// NOLINTEND

void fillInterface([[maybe_unused]] cesium::omniverse::tests::CesiumOmniverseCppTestsPlugin& iface) {}
