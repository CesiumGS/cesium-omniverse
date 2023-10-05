#include "UsdUtilTests.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/LoggerSink.h"
#include "cesium/omniverse/UsdUtil.h"

#include <doctest/doctest.h>

void runAllUsdUtilTests() {
    // CESIUM_LOG_INFO macro can only be run inside the cesium::omniverse namespace
    using namespace cesium::omniverse;
    CESIUM_LOG_INFO("Running UsdUtil Tests...");
    CHECK(UsdUtil::primExists(pxr::SdfPath("/Cesium")));
    CESIUM_LOG_INFO("UsdUtil Tests complete!");
}
