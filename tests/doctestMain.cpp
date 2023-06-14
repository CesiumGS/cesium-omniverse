#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#define DOCTEST_CONFIG_NO_EXCEPTIONS
#define DOCTEST_CONFIG_IMPLEMENT

#include "testUtils.h"

#include <doctest/doctest.h>

#include <filesystem>

int main(int argc, char** argv) {
    doctest::Context context;

    context.applyCommandLine(argc, argv);

    // Some tests contain relative paths rooted in the top level project dir
    // so we set this as the working directory
    std::filesystem::path oldWorkingDir = std::filesystem::current_path();
    std::filesystem::current_path(TEST_WORKING_DIRECTORY);

    int res = context.run(); // run

    // restore the previous working directory
    std::filesystem::current_path(oldWorkingDir);

    if (context.shouldExit()) // important - query flags (and --exit) rely on the user doing this
        return res;           // propagate the result of the tests

    int client_stuff_return_code = 0;
    // your program - if the testing framework is integrated in your production code

    return res + client_stuff_return_code; // the result from doctest is propagated here as well
}
