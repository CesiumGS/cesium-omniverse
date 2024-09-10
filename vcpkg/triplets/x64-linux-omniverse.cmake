include("${CMAKE_CURRENT_LIST_DIR}/shared/common.cmake")

set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)

set(VCPKG_CMAKE_SYSTEM_NAME Linux)

# Build with old C++ ABI. See top-level CMakeLists.txt for explanation.
set(VCPKG_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")
set(VCPKG_C_FLAGS "${VCPKG_CXX_FLAGS}")
