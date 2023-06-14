# This file is separate from the root CMakeLists.txt and ConfigureConan.cmake
# since the hash of this file is used as a cache key on CI

include(ConfigureConan)

set(REQUIRES
    "cpr/1.9.0@#168b3dcd0f24fc690873881d11ac1e66"
    "doctest/2.4.9@#ea6440e3cd544c9a25bf3a96bcf16f48"
    "openssl/1.1.1s@#c6838ae653d103956aec228f7642c45a"
    "pybind11/2.10.1@#561736204506dad955276aaab438aab4"
    "stb/cci.20220909@#1c47474f095ef8cd9e4959558525b827"
    "zlib/1.2.13@#13c96f538b52e1600c40b88994de240f"
    "yaml-cpp/0.7.0@#85b409c274a53d226b71f1bdb9cb4f8b"
    "libcurl/7.86.0@#88506b3234d553b90af1ceefc3dd1652"
    "nasm/2.15.05@#799d63b1672a337584b09635b0f22fc1")

if(WIN32)
    set(REQUIRES ${REQUIRES} "strawberryperl/5.32.1.1@#8f83d05a60363a422f9033e52d106b47")
endif()

# cmake-format: off
configure_conan(
    PROJECT_BUILD_DIRECTORY
        "${PROJECT_BINARY_DIR}"
    REQUIRES
        ${REQUIRES}
)
# cmake-format: on
