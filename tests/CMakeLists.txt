add_subdirectory(bindings)
add_subdirectory(src)
add_subdirectory(include)

add_test(NAME tests_extension COMMAND extern/nvidia/_build/target-deps/kit-sdk/kit
                                      ./apps/cesium.omniverse.cpp.tests.runner.kit)
