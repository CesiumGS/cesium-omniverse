# This file is separate from the root CMakeLists.txt and ConfigureConan.cmake
# since the hash of this file is used as a cache key on CI

include(ConfigureConan)

set(REQUIRES
    "cpr/1.10.4@#860d6cd6d8eb5f893e647c2fb016eb61"
    "doctest/2.4.9@#ea6440e3cd544c9a25bf3a96bcf16f48"
    "openssl/1.1.1w@#42c32b02f62aa987a58201f4c4561d3e"
    "pybind11/2.10.1@#561736204506dad955276aaab438aab4"
    "stb/cci.20220909@#1c47474f095ef8cd9e4959558525b827"
    "zlib/1.2.13@#13c96f538b52e1600c40b88994de240f"
    "yaml-cpp/0.7.0@#85b409c274a53d226b71f1bdb9cb4f8b"
    "libcurl/8.2.1@#8f62ba7135f5445e5fe6c4bd85143b53"
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
