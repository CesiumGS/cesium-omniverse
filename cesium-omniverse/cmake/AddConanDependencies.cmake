# This file is separate from the root CMakeLists.txt and ConfigureConan.cmake
# since the hash of this file is used as a cache key on CI

include(ConfigureConan)

# cmake-format: off
configure_conan(
    PROJECT_BUILD_DIRECTORY
        "${PROJECT_BINARY_DIR}"
    REQUIRES
        cpr/1.9.0
        doctest/2.4.9
        pybind11/2.10.0
        zlib/1.2.13
)
# cmake-format: on
