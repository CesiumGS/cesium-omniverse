#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/LoggerSink.h"
#include "cesium/omniverse/UsdUtil.h"

#include <doctest/doctest.h>

TEST_SUITE("UsdUtil tests") {
    TEST_CASE("Check expected initial state") {
        CHECK(cesium::omniverse::UsdUtil::primExists(pxr::SdfPath("/Cesium")));
    }
}
