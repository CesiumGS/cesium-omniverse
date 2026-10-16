# This file is separate from the root CMakeLists.txt and ConfigureConan.cmake
# since the hash of this file is used as a cache key on CI

include(ConfigureConan)

set(REQUIRES
    "doctest/2.4.9@#ea6440e3cd544c9a25bf3a96bcf16f48"
    "pybind11/2.10.1@#561736204506dad955276aaab438aab4"
    "yaml-cpp/0.7.0@#85b409c274a53d226b71f1bdb9cb4f8b")

# cmake-format: off
configure_conan(
    PROJECT_BUILD_DIRECTORY
        "${PROJECT_BINARY_DIR}"
    REQUIRES
        ${REQUIRES}
)
# cmake-format: on
