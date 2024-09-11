include("${CMAKE_CURRENT_LIST_DIR}/shared/common.cmake")

set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)

# Techically we should be setting /MDd if USE_NVIDIA_RELEASE_LIBRARIES is false
set(VCPKG_CXX_FLAGS "/MD /MP /EHsc")
set(VCPKG_C_FLAGS "${VCPKG_CXX_FLAGS}")
