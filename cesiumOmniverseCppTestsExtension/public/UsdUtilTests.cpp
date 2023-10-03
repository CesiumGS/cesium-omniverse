// #define DOCTEST_CONFIG_IMPLEMENT
// #define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
// #define DOCTEST_CONFIG_SUPER_FAST_ASSERTS

#include "UsdUtilTests.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/LoggerSink.h"
#include "cesium/omniverse/UsdUtil.h"

#include <doctest/doctest.h>

#include <iostream>

void run_all_UsdUtil_tests() {
    using namespace cesium::omniverse;
    CESIUM_LOG_INFO("Running UsdUtil Tests...");
    CHECK(cesium::omniverse::UsdUtil::primExists(pxr::SdfPath("/Cesium")));
    CESIUM_LOG_INFO("UsdUtil Tests complete!");
}
