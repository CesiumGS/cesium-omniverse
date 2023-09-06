#include <carb/BindingsPythonUtils.h>
#include "CesiumTests.h"

// NOLINTNEXTLINE
CARB_BINDINGS("cesium.tests.python")
DISABLE_PYBIND11_DYNAMIC_CAST(cesium::omniverse::tests::ICesiumOmniverseTestsInterface)

PYBIND11_MODULE(CesiumOmniverseTestsPythonBindings, m) {

    using namespace cesium::omniverse::tests;

    m.doc() = "pybind11 cesium.omniverse.tests bindings";

    // clang-format off
    carb::defineInterfaceClass<ICesiumOmniverseTestsInterface>(
        m, "ICesiumOmniverseTestsInterface", "acquire_cesium_omniverse_tests_interface", "release_cesium_omniverse_tests_interface")
        .def("run_all_tests", &ICesiumOmniverseTestsInterface::run_all_tests);
    // clang-format on

}
