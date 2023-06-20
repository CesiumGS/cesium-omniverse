#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#define DOCTEST_CONFIG_NO_EXCEPTIONS

// Defining this magic variable and importing doctest are all that's required
// Doctest will automatically find all the TEST_CASEs linked/defined elsewhere
// and include them in a main function it builds
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest/doctest.h>
