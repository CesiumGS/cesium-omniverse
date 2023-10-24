# Testing Guide

## Python Tests
Python tests are run through `pytest` (see full documentation [here](https://docs.pytest.org/en/latest/)). To run these tests with the proper sourcing and environment, simpy run `scripts/run_python_unit_tests.(bat|sh)`. You can also run these
tests via the app. Open the extensions window while running omniverse. Find and select the Cesium for Omniverse Extension, then navigate to its Tests tab. The "Run Extension Tests" button will run the python tests (not the C++ tests).

## C++ Tests (The Tests Extension)
C++ tests are run through `doctest`, which is set up and run via the Tests Extension.
Normally `doctest` can be run via the command line, but since much of the code we test
can only run properly inside omniverse, we run the tests there too.
The easiest way to run the tests extension is via the launch configuration in vs code. Simply go to the `run and debug` dropdown and launch the `Tests Extension`. The testing output is provided in the terminal used to launch everything. Failed tests will be caught by the debugger, though you may need to go one level up in the execution stack to see the `CHECK` being called.

To run the extension via the command line, simply pass the tests extension's kit config file to kit with `extern/nvidia/_build/target-deps/kit-sdk/kit ./apps/cesium.omniverse.cpp.tests.runner.kit`

[doctest documentation](https://bit.ly/doctest-docs) can be found here.

## How do I add a new test?
### Python
`pytest` will auto-discover functions matching the pattern `test_.*` (and other patterns).
If you want your tests to be included in the tests for the main extension, import it into `exts/cesium.omniverse/cesium/omniverse/tests/__init__.py`.
See [extension_test.py](../../exts/cesium.omniverse/cesium/omniverse/tests/extension_test.py) for an example

### C++
`TEST_SUITE`s and `TEST_CASE`s defined in `tests/src/*.cpp` will be auto-discovered by the `run_all_tests` function in `tests/src/CesiumOmniverseCppTests.cpp`. These macros perform some automagic function definitions, so they are best left outside of other function/class definitions. See `tests/src/ExampleTests.cpp` for examples of basic tests and more advanced use cases, such as using a config file to
define expected outputs or parameterizing tests.
