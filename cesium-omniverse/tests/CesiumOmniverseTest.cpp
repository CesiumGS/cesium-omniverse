#include "cesium/omniverse/CesiumOmniverse.h"

#include <doctest/doctest.h>

TEST_SUITE("Test") {
    TEST_CASE("getNumber") {
        CHECK(getNumber() == 2);
    }
}
