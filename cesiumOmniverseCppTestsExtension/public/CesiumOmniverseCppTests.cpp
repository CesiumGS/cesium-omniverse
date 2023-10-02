#define CARB_EXPORTS
#define DOCTEST_CONFIG_IMPLEMENT
#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS

#include "CesiumOmniverseCppTests.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/LoggerSink.h"
#include "cesium/omniverse/UsdUtil.h"

#include <carb/PluginUtils.h>
#include <doctest/doctest.h>
#include <omni/fabric/IFabric.h>

#include <iostream>

static void exampleTest() {
    // This file is set up such that we can use doctest assertions (e.x. CHECK)
    // outside of a test suite.

    // NOTE: failed tests will throw a SIGTRAP in order to call the handler
    // defined below, which will trigger a pause in any running debugger.
    // Execution can continue normally from there. To see this in action,
    // modify the CHECK below to fail then run the extension in a debugger

    CHECK(1 + 1 == 2);
}

static void handler(const doctest::AssertData& ad) {
    // Handles failed doctest assertions

    using namespace doctest;

    std::cout << Color::LightGrey << skipPathFromFilename(ad.m_file) << "(" << ad.m_line << "): ";
    std::cout << Color::Red << failureString(ad.m_at) << ": ";

    // handling only normal (comparison and unary) asserts - exceptions-related asserts have been skipped
    if (ad.m_at & assertType::is_normal) {
        std::cout << Color::Cyan << assertString(ad.m_at) << "( " << ad.m_expr << " ) ";
        std::cout << Color::None << (ad.m_threw ? "THREW exception: " : "is NOT correct!\n");
        if (ad.m_threw)
            std::cout << ad.m_exception;
        else
            std::cout << "  values: " << assertString(ad.m_at) << "( " << ad.m_decomp << " )";
    } else {
        std::cout << Color::None << "an assert dealing with exceptions has failed!";
    }

    std::cout << std::endl;
}

namespace cesium::omniverse::tests {

class CesiumOmniverseCppTestsPlugin final : public ICesiumOmniverseCppTestsInterface {
  public:
    void onStartup(const char* cesiumExtensionLocation, const char* kitVersion) noexcept override {
        Context::onStartup(cesiumExtensionLocation, kitVersion);
    }

    void onShutdown() noexcept override {
        Context::onShutdown();
    }

    void run_all_tests(long int stage_id) noexcept override {

        CESIUM_LOG_INFO("Running Cesium Omniverse Tests with stage id: {}", stage_id);

        // construct a context
        doctest::Context context;

        // sets the context as the default one - so asserts used outside of a testing context do not crash
        context.setAsDefaultForAssertsOutOfTestCases();

        // set a handler with a signature: void(const doctest::AssertData&)
        // without setting a handler we would get std::abort() called when an assert fails
        context.setAssertHandler(handler);

        exampleTest();

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
CARB_PLUGIN_IMPL_DEPS(omni::fabric::IFabric, omni::fabric::IStageReaderWriter)
// NOLINTEND

void fillInterface([[maybe_unused]] cesium::omniverse::tests::CesiumOmniverseCppTestsPlugin& iface) {}
