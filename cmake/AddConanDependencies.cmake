# This file is separate from the root CMakeLists.txt and ConfigureConan.cmake
# since the hash of this file is used as a cache key on CI

include(ConfigureConan)

set(REQUIRES
    "doctest/2.4.9@#ea6440e3cd544c9a25bf3a96bcf16f48"
    "pybind11/2.10.1@#561736204506dad955276aaab438aab4"
    "stb/cci.20220909@#1c47474f095ef8cd9e4959558525b827"
    "zlib/1.2.13@#13c96f538b52e1600c40b88994de240f"
    "yaml-cpp/0.7.0@#85b409c274a53d226b71f1bdb9cb4f8b"
    "ms-gsl/4.1.0@#e2af591b5e8c3a3f6a570cf9cbcd7a87")

# cmake-format: off
configure_conan(
    PROJECT_BUILD_DIRECTORY
        "${PROJECT_BINARY_DIR}"
    REQUIRES
        ${REQUIRES}
)
# cmake-format: on
