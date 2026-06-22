include("${CMAKE_CURRENT_LIST_DIR}/shared/common.cmake")

set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)

# Techically we should be setting /MDd if USE_NVIDIA_RELEASE_LIBRARIES is false
set(VCPKG_CXX_FLAGS "/MD /EHsc -D_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR")
set(VCPKG_C_FLAGS "${VCPKG_CXX_FLAGS}")

# CMake 4.x removed compatibility with cmake_minimum_required(<3.5); some ports
# (e.g. asyncplusplus) still declare it. Allow them to configure anyway.
set(VCPKG_CMAKE_CONFIGURE_OPTIONS "-DCMAKE_POLICY_VERSION_MINIMUM=3.5")

