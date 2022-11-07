include(Macros)

glob_files(SOURCES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

# TODO: maybe PythonBindings and InMemoryAssetResolver should be separate sub-folders since they are conceptually different targets
set(PYTHON_BINDINGS_CPP "${CMAKE_CURRENT_LIST_DIR}/PythonBindings.cpp")
set(IN_MEMORY_ASSET_RESOLVER_CPP "${CMAKE_CURRENT_LIST_DIR}/InMemoryAssetResolver.cpp")
list(
    REMOVE_ITEM
    SOURCES
    ${PYTHON_BINDINGS_CPP}
    ${IN_MEMORY_ASSET_RESOLVER_CPP})

set(IN_MEMORY_ASSET_RESOLVER_CXX_DEFINES ${CESIUM_OMNI_CXX_DEFINES} CESIUM_OMNI_EXPORTS)
set(CESIUM_OMNIVERSE_LIB_CXX_DEFINES ${CESIUM_OMNI_CXX_DEFINES} CESIUM_OMNI_EXPORTS)
set(CESIUM_OMNIVERSE_PYTHON_BINDINGS_CXX_DEFINES ${CESIUM_OMNI_CXX_DEFINES})

if(NOT BUILD_SHARED_LIBS)
    set(CESIUM_OMNIVERSE_LIB_CXX_DEFINES ${CESIUM_OMNIVERSE_LIB_CXX_DEFINES} CESIUM_OMNI_STATIC)
    set(CESIUM_OMNIVERSE_PYTHON_BINDINGS_CXX_DEFINES ${CESIUM_OMNIVERSE_PYTHON_BINDINGS_CXX_DEFINES} CESIUM_OMNI_STATIC)
endif()

# On Linux you may see a warning like:
#
# Warning: Unused direct dependencies:
#         cesium-omniverse/extern/nvidia/_build/target-deps/usd/release/lib/libarch.so
#         cesium-omniverse/extern/nvidia/_build/target-deps/usd/release/lib/libjs.so
#         cesium-omniverse/extern/nvidia/_build/target-deps/usd/release/lib/libkind.so
#         cesium-omniverse/extern/nvidia/_build/target-deps/usd/release/lib/libndr.so
#         cesium-omniverse/extern/nvidia/_build/target-deps/usd/release/lib/libpcp.so
#         cesium-omniverse/extern/nvidia/_build/target-deps/usd/release/lib/libplug.so
#         cesium-omniverse/extern/nvidia/_build/target-deps/usd/release/lib/libsdr.so
#         cesium-omniverse/extern/nvidia/_build/target-deps/usd/release/lib/libtrace.so
#
# This is because some USD libraries are referenced by other USD libraries but are not direct depenencies of
# Cesium Omniverse. The reason we include them in the list below is because on Windows this allows them to be
# copied to the bin folder when TARGET_RUNTIME_DLLS is used.
#
#
# You may also see warnings like:
#
# CMake Warning at build/src/cmake_install.cmake:115 (file):
#   Dependency libpython3.7m.so.1.0 found in search directory:
#     cesium-omniverse/extern/nvidia/_build/target-deps/python/lib
#
#   See file(GET_RUNTIME_DEPENDENCIES) documentation for more information.
# Call Stack (most recent call first):
#   build/cmake_install.cmake:52 (include)
#
# This is because the USD libraries .so rpath points to a non-existent python path
# E.g. `readelf -d libusd.so` will show `$ORIGIN/../../../_dependencies/python3/lib`
# This path doesn't exist, so in order to find python we need to explicility add it
# to the search path. Unfortunately CMake will print a warning when python .so is
# found this way.

# cmake-format: off
setup_lib(
    TARGET_NAME
        InMemoryAssetResolver
    TYPE
        SHARED # InMemoryAssetResolver needs to be built as a SHARED library always
    SOURCES
        "${IN_MEMORY_ASSET_RESOLVER_CPP}"
    INCLUDE_DIRS
        "${PROJECT_SOURCE_DIR}/include"
    LIBRARIES
        ar
        arch
        usd
        tbb
        tf
    CXX_FLAGS
        ${CESIUM_OMNI_CXX_FLAGS}
    CXX_FLAGS_DEBUG
        ${CESIUM_OMNI_CXX_FLAGS_DEBUG}
    CXX_DEFINES
        ${IN_MEMORY_ASSET_RESOLVER_CXX_DEFINES}
    CXX_DEFINES_DEBUG
        ${CESIUM_OMNI_CXX_DEFINES_DEBUG}
)
# cmake-format: on

# cmake-format: off
setup_lib(
    TARGET_NAME
        CesiumOmniverse
    SOURCES
        ${SOURCES}
    INCLUDE_DIRS
        "${PROJECT_SOURCE_DIR}/include"
    LIBRARIES
        InMemoryAssetResolver
        Cesium3DTilesSelection
        CesiumAsync
        CesiumGeometry
        CesiumGeospatial
        CesiumGltf
        CesiumGltfReader
        CesiumJsonReader
        CesiumUtility
        async++
        draco
        ktx_read
        modp_b64
        s2geometry
        spdlog
        uriparser
        webpdecoder
        cpr::cpr
        stb::stb
        ZLIB::ZLIB
        boost_python37
        python37
        tbb
        ar
        arch
        gf
        js
        kind
        ndr
        pcp
        plug
        sdf
        sdr
        tf
        trace
        usd
        usdGeom
        usdShade
        usdUtils
        vt
        work
    DEPENDENCIES
        cesium-native-external
    CXX_FLAGS
        ${CESIUM_OMNI_CXX_FLAGS}
    CXX_FLAGS_DEBUG
        ${CESIUM_OMNI_CXX_FLAGS_DEBUG}
    CXX_DEFINES
        ${CESIUM_OMNIVERSE_LIB_CXX_DEFINES}
    CXX_DEFINES_DEBUG
        ${CESIUM_OMNI_CXX_DEFINES_DEBUG}
)
# cmake-format: on

# cmake-format: off
setup_python_module(
    TARGET_NAME
        CesiumOmniversePythonBindings
    SOURCES
        "${PYTHON_BINDINGS_CPP}"
    LIBRARIES
        CesiumOmniverse
        pybind11::pybind11
    CXX_FLAGS
        ${CESIUM_OMNI_CXX_FLAGS}
    CXX_FLAGS_DEBUG
        ${CESIUM_OMNI_CXX_FLAGS_DEBUG}
    CXX_DEFINES
        ${CESIUM_OMNIVERSE_PYTHON_BINDINGS_CXX_DEFINES}
    CXX_DEFINES_DEBUG
        ${CESIUM_OMNI_CXX_DEFINES_DEBUG}
)
# cmake-format: on

# Copy plugInfo.json and mem.cesium into each build folder
add_custom_command(
    TARGET InMemoryAssetResolver
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_BINARY_DIR}/plugInfo.json.build"
            "$<TARGET_FILE_DIR:InMemoryAssetResolver>/plugins/InMemoryAssetResolver/resources/plugInfo.json"
    COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_BINARY_DIR}/mem.cesium"
            "$<TARGET_FILE_DIR:InMemoryAssetResolver>/mem.cesium")
