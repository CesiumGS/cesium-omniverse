#include "CesiumOmniverseTests.h"

#include <carb/BindingsPythonUtils.h>

// NOLINTNEXTLINE
CARB_BINDINGS("cesium.omniverse.tests.python")
DISABLE_PYBIND11_DYNAMIC_CAST(cesium::omniverse::tests::ICesiumOmniverseTestsInterface)

PYBIND11_MODULE(CesiumOmniverseTestsPythonBindings, m) {

    using namespace cesium::omniverse::tests;

    m.doc() = "pybind11 cesium.omniverse.tests bindings";

    // clang-format off
    carb::defineInterfaceClass<ICesiumOmniverseTestsInterface>(
        m, "ICesiumOmniverseTestsInterface", "acquire_cesium_omniverse_tests_interface", "release_cesium_omniverse_tests_interface")
        .def("run_all_tests", &ICesiumOmniverseTestsInterface::run_all_tests)
        .def("on_startup", &ICesiumOmniverseTestsInterface::onStartup)
        .def("on_shutdown", &ICesiumOmniverseTestsInterface::onShutdown);
    // clang-format on
}
