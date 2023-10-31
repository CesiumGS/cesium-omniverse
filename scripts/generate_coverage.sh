#! /bin/bash

# make sure we're in the root dir of the repo
cd `dirname "${BASH_SOURCE[0]}"`/..

echo "Building with coverage enabled"
cmake -B build-debug -D CMAKE_BUILD_TYPE=Debug -D CESIUM_OMNI_ENABLE_COVERAGE=true
cmake --build build-debug --target install --parallel 8
# reset the debug build to have coverage disabled, otherwise future builds will still have it
cmake -B build-debug -D CMAKE_BUILD_TYPE=Debug -D CESIUM_OMNI_ENABLE_COVERAGE=false

echo "removing old coverage files"
find . -name '*.gcda' -delete

echo "Starting tests extension, please exit manually any time after omniverse has fully loaded"
./extern/nvidia/_build/target-deps/kit-sdk/kit ./apps/cesium.omniverse.cpp.tests.runner.kit > /dev/null

echo "Gathering coverage data"

# delete GCDA files after processing
# print text output
# generate HTML report with annotated source code
mkdir -p coverage
gcovr --delete --txt --html-details "coverage/index.html" --filter=src --filter=include

echo "opening report"
open coverage/index.html
