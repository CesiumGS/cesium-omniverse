# This file is separate from the root CMakeLists.txt and ConfigureConan.cmake
# since the hash of this file is used as a cache key on CI

include(ConfigureConan)

set(REQUIRES
    "doctest/2.4.11@#a4211dfc329a16ba9f280f9574025659"
    "openssl/3.2.1@#39bd48ed31f1f3fbdcd75a0648aaedcf"
    "pybind11/2.11.1@#e24cefefdb5561ba8d8bc34ab5ba1607"
    "zlib/1.3.1@#f52e03ae3d251dec704634230cd806a2"
    "yaml-cpp/0.8.0@#720ad361689101a838b2c703a49e9c26"
    "libcurl/8.6.0@#ff220b1555b8aebbb78440b25d471217")

# cmake-format: off
configure_conan(
    PROJECT_BUILD_DIRECTORY
        "${PROJECT_BINARY_DIR}"
    REQUIRES
        ${REQUIRES}
)
# cmake-format: on
