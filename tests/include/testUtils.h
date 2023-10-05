#pragma once

// ************************************
// ***** Before editing this file *****
// ************************************
// be sure you are editing testUtils.h.in and not directly editing testUtils.h
// cmake generates testUtils.h at build time by processing testUtils.h.in
// in order to perform some string replacement

#include <algorithm>
#include <cstdlib>
#include <list>
#include <string>

#include <yaml-cpp/yaml.h>

#define NUM_TEST_REPETITIONS 100
// Some tests contain relative paths rooted in the top level project dir
// cmake fills in this information via string replacement
#define TEST_WORKING_DIRECTORY "/home/melser/git/cesium-omniverse"

// Macro for parameterizing test data using one currently recommended method.
// See the doctest docs for more info:
// https://github.com/doctest/doctest/blob/ae7a13539fb71f270b87eb2e874fbac80bc8dda2/doc/markdown/parameterized-tests.md

#define DOCTEST_VALUE_PARAMETERIZED_DATA(data, data_container)                                                       \
    static size_t _doctest_subcase_idx = 0;                                                                          \
    std::for_each(data_container.begin(), data_container.end(), [&](const auto& in) {                                \
        DOCTEST_SUBCASE((std::string(#data_container "[") + std::to_string(_doctest_subcase_idx++) + "]").c_str()) { \
            data = in;                                                                                               \
        }                                                                                                            \
    });                                                                                                              \
    _doctest_subcase_idx = 0

void fillWithRandomInts(std::list<int>& lst, int min, int max, int n);

using ConfigMap = YAML::Node;
ConfigMap getScenarioConfig(const std::string& scenario, YAML::Node configRoot);
