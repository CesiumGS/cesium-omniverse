#include "CesiumOmniverseCppTests.h"

#include <carb/BindingsPythonUtils.h>

#ifdef CESIUM_OMNI_CLANG
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif

CARB_BINDINGS("cesium.omniverse.cpp.tests.python")

#ifdef CESIUM_OMNI_CLANG
#pragma clang diagnostic pop
#endif

DISABLE_PYBIND11_DYNAMIC_CAST(cesium::omniverse::tests::ICesiumOmniverseCppTestsInterface)

PYBIND11_MODULE(CesiumOmniverseCppTestsPythonBindings, m) {

    using namespace cesium::omniverse::tests;

    m.doc() = "pybind11 cesium.omniverse.cpp.tests bindings";

    // clang-format off
    carb::defineInterfaceClass<ICesiumOmniverseCppTestsInterface>(
        m, "ICesiumOmniverseCppTestsInterface", "acquire_cesium_omniverse_tests_interface", "release_cesium_omniverse_tests_interface")
        .def("set_up_tests", &ICesiumOmniverseCppTestsInterface::setUpTests)
        .def("run_all_tests", &ICesiumOmniverseCppTestsInterface::runAllTests)
        .def("on_startup", &ICesiumOmniverseCppTestsInterface::onStartup)
        .def("on_shutdown", &ICesiumOmniverseCppTestsInterface::onShutdown);
    // clang-format on
}
