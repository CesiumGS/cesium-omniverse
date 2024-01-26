#define CARB_EXPORTS
#define DOCTEST_CONFIG_IMPLEMENT
#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS

#include "CesiumOmniverseCppTests.h"

#include "UsdUtilTests.h"
#include "testUtils.h"
#include "tilesetTests.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/Logger.h"

#include <carb/PluginUtils.h>
#include <cesium/omniverse/UsdUtil.h>
#include <doctest/doctest.h>
#include <omni/fabric/IFabric.h>
#include <omni/kit/IApp.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/stage.h>

#include <iostream>

namespace cesium::omniverse::tests {

class CesiumOmniverseCppTestsPlugin final : public ICesiumOmniverseCppTestsInterface {
  public:
    void onStartup(const char* cesiumExtensionLocation) noexcept override {
        _pContext = std::make_unique<Context>(cesiumExtensionLocation);
    }

    void onShutdown() noexcept override {
        _pContext = nullptr;
    }

    void setUpTests(long int stage_id) noexcept override {
        // This runs after the stage has been created, but at least one frame
        // before runAllTests. This is to allow time for USD notifications to
        // propogate, as prims cannot be created and used on the same frame.

        _pContext->getLogger()->info("Setting up Cesium Omniverse Tests with stage id: {}", stage_id);

        _pContext->onUsdStageChanged(stage_id);

        auto rootPath = cesium::omniverse::UsdUtil::getRootPath(_pContext->getUsdStage());

        setUpUsdUtilTests(_pContext.get(), rootPath);
        setUpTilesetTests(_pContext.get(), rootPath);
    }

    void runAllTests() noexcept override {
        _pContext->getLogger()->info("Running Cesium Omniverse Tests");

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

        _pContext->getLogger()->info("Cesium Omniverse tests complete");

        _pContext->getLogger()->info("Cleaning up after tests");
        cleanUpAfterTests();
        _pContext->getLogger()->info("Cesium Omniverse test prims removed");
    }

    void cleanUpAfterTests() noexcept {
        // delete any test related prims here
        auto pUsdStage = _pContext->getUsdStage();
        cleanUpUsdUtilTests(pUsdStage);
        cleanUpTilesetTests(pUsdStage);
    }

  private:
    std::unique_ptr<Context> _pContext;
};

} // namespace cesium::omniverse::tests

const struct carb::PluginImplDesc pluginImplDesc = {
    "cesium.omniverse.cpp.tests.plugin",
    "Cesium Omniverse Tests Plugin.",
    "Cesium",
    carb::PluginHotReload::eDisabled,
    "dev"};

#ifdef CESIUM_OMNI_CLANG
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif

CARB_PLUGIN_IMPL(pluginImplDesc, cesium::omniverse::tests::CesiumOmniverseCppTestsPlugin)
CARB_PLUGIN_IMPL_DEPS(omni::fabric::IFabric, omni::kit::IApp, carb::settings::ISettings)

#ifdef CESIUM_OMNI_CLANG
#pragma clang diagnostic pop
#endif

void fillInterface([[maybe_unused]] cesium::omniverse::tests::CesiumOmniverseCppTestsPlugin& iface) {}
