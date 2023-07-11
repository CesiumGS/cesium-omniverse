/*
* A collection of simple tests to demonstrate Doctest
*/

#include "testUtils.h"

#include <doctest/doctest.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <list>
#include <stdexcept>
#include <vector>

#include <yaml-cpp/yaml.h>

const std::string CONFIG_PATH = "tests/configs/exampleConfig.yaml";

// Test Suites are not required, but this sort of grouping makes it possible
// to select which tests do/don't run via command line options
TEST_SUITE("Example Tests") {
    // ----------------------------------------------
    //                   Basic Tests
    // ----------------------------------------------

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

    // ----------------------------------------------
    //             YAML Config Examples
    // ----------------------------------------------

    std::string transmogrifier(const std::string& s) {
        // an example function with differing output for some scenarios
        if (s == "scenario2") {
            return "bar";
        }
        return "foo";
    }

    void checkAgainstExpectedResults(const std::string& scenarioName, const YAML::Node& expectedResults) {

        // we have to specify the type of the desired data from the config via as()
        CHECK(3.14159 == expectedResults["pi"].as<double>());
        CHECK(2 == expectedResults["onlyEvenPrime"].as<int>());

        // as() does work for some non-scalar types, such as vectors, lists, and maps
        // for adding custom types to the config, see:
        // https://github.com/jbeder/yaml-cpp/wiki/Tutorial#converting-tofrom-native-data-types
        auto fib = expectedResults["fibonacciSeq"].as<std::vector<int>>();
        CHECK(fib[2] + fib[3] == fib[4]);

        // More complicated checks can be done with helper functions that take the scenario as input
        CHECK(transmogrifier(scenarioName) == expectedResults["transmogrifierOutput"].as<std::string>());
    }

    TEST_CASE("Use a config file to detail multiple scenarios") {

        YAML::Node configRoot = YAML::LoadFile(CONFIG_PATH);

        // The config file has default parameters and
        // an override for one or more scenarios
        std::vector<std::string> scenarios = {"scenario1", "scenario2", "scenario3"};

        for (const auto& s : scenarios) {
            ConfigMap conf = getScenarioConfig(s, configRoot);
            checkAgainstExpectedResults(s, conf);
        }
    }

    // ----------------------------------------------
    //                   Misc.
    // ----------------------------------------------

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
