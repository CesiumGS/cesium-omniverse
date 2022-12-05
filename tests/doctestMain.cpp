#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#define DOCTEST_CONFIG_IMPLEMENT
#define DOCTEST_CONFIG_NO_EXCEPTIONS

#include <doctest/doctest.h>

#include <cstdio>

int main(int argc, char* argv[]) {
    // Initialize doctest
    doctest::Context context;
    context.applyCommandLine(argc, argv);

    // Run tests
    const int result = context.run();

    return result;
}
