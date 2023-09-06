#define CARB_EXPORTS

#include "CesiumTests.h"

#include "cesium/omniverse/Context.h"

#include <carb/PluginUtils.h>
#include <omni/fabric/IFabric.h>

#include <iostream>

namespace cesium::omniverse::tests {

class CesiumOmniverseTestsPlugin final : public ICesiumOmniverseTestsInterface {
  public:
    void onStartup(const char* cesiumExtensionLocation) noexcept override {
        Context::onStartup(cesiumExtensionLocation);
    }

    void onShutdown() noexcept override {
        Context::onShutdown();
    }

    void run_all_tests(long int stage_id) noexcept override {

        std::cout << "Running Cesium Omniverse Tests with stage id: " << stage_id << std::endl;
        // TODO run tests

        std::cout << "Cesium Omniverse Tests complete" << std::endl;
    }
};

} // namespace cesium::omniverse::tests

const struct carb::PluginImplDesc pluginImplDesc =
    {"cesium.tests.plugin", "Cesium Omniverse Tests Plugin.", "Cesium", carb::PluginHotReload::eDisabled, "dev"};

// NOLINTBEGIN
CARB_PLUGIN_IMPL(pluginImplDesc, cesium::omniverse::tests::CesiumOmniverseTestsPlugin)
CARB_PLUGIN_IMPL_DEPS(omni::fabric::IFabric, omni::fabric::IStageReaderWriter)
// NOLINTEND

void fillInterface([[maybe_unused]] cesium::omniverse::tests::CesiumOmniverseTestsPlugin& iface) {}
