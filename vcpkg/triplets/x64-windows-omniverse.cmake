include("${CMAKE_CURRENT_LIST_DIR}/shared/common.cmake")

set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)

set(VCPKG_CMAKE_CONFIGURE_OPTIONS "")

# CMake 4.x removed compatibility with cmake_minimum_required(<3.5); some ports
# (e.g. asyncplusplus) still declare it. Allow them to configure anyway.
list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS "-DCMAKE_POLICY_VERSION_MINIMUM=3.5")

# Kit SDK links against the release MSVC Runtime Library so we
# need to force vcpkg dependencies to do the same.
set(VCPKG_POLICY_ONLY_RELEASE_CRT enabled)
list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS "-DCMAKE_MSVC_RUNTIME_LIBRARY:STRING=MultiThreadedDLL")
list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS "-DCMAKE_CXX_FLAGS_DEBUG:STRING=/MD")
list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS "-DCMAKE_C_FLAGS_DEBUG:STRING=/MD")

# Uncomment these lines and comment out the lines above if USE_NVIDIA_RELEASE_LIBRARIES is false
# list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS "-DCMAKE_MSVC_RUNTIME_LIBRARY:STRING=MultiThreadedDebugDLL")
# list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS "-DCMAKE_CXX_FLAGS_DEBUG:STRING=/MDd")
# list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS "-DCMAKE_C_FLAGS_DEBUG:STRING=/MDd")

# Kit SDK is built with the VS 2019 C++ compiler so we need
# to build vcpkg dependencies with the same version
# See extern/nvidia/_build/target-deps/kit-sdk/dev/premake5-public.lua
set(VCPKG_PLATFORM_TOOLSET_VERSION "14.29")
