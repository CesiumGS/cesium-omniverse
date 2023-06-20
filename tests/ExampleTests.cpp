/*
* A collection of simple tests to demonstrate Doctest
*/

#include "doctestUtils.h"

#include <doctest/doctest.h>

#include <cstdint>
#include <list>
#include <stdexcept>

// Test Suites are not required, but this sort of grouping makes it possible
// to select which tests do/don't run via command line options
TEST_SUITE("Example Tests") {

    TEST_CASE("The most basic test") {
        CHECK(1 + 1 == 2);
    }

    TEST_CASE("Demonstrating Subcases") {
        // This initialization is shared between all subcases
        int x = 1;

        // Note that these two subcases run independantly of each other!
        SUBCASE("Increment") {
            x += 1;
            CHECK(x == 2);
        }
        SUBCASE("Decrement") {
            x -= 1;
            CHECK(x == 0);
        }
    }
    // A few notes on subcases:
    //  - You can nest subcases
    //  - Subcases work by creating multiple calls to the higher level case,
    //    where each call proceeds to only one of the subcases. If you generate
    //    excessive subcases, watch out for a stack overflow.

    void runPositiveCheck(int64_t val) {
        // helper function for parameterized test method 1
        CHECK(val > 0);
    }

    TEST_CASE("Demonstrate Parameterized Tests - method 1") {
        // Generate the data you want the tests to iterate over
        std::list<uint32_t> dataContainer = {42, 64, 8675309, 1024};

        for (auto i : dataContainer) {
            CAPTURE(i);
            runPositiveCheck(i);
        }
    }

    TEST_CASE("Demonstrate Parameterized Tests - method 2") {
        // Generate the data you want the tests to iterate over
        uint32_t item;
        std::list<uint32_t> dataContainer = {42, 64, 8675309, 1024};

        // This macro from doctestUtils.h will generate a subcase per datum
        DOCTEST_VALUE_PARAMETERIZED_DATA(item, dataContainer);

        // this check will now be run for each datum
        CHECK(item > 0);
    }

    TEST_CASE("A few other useful macros") {
        // The most common test macro is CHECK, but others are available
        // Here are just a few

        // Any failures here will prevent the rest of the test from running
        REQUIRE(0 == 0);

        // Make sure the enclosed code does/doesn't throw an exception
        CHECK_THROWS(throw "test exception!");
        CHECK_NOTHROW(if (false) throw "should not throw");

        // Prints a warning if the assert fails, but does not fail the test
        WARN(true);
    }
}
