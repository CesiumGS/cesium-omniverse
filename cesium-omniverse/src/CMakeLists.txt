include(Macros)

glob_files(SOURCES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
set(PYTHON_BINDINGS_CPP "${CMAKE_CURRENT_LIST_DIR}/PythonBindings.cpp")
list(REMOVE_ITEM SOURCES ${PYTHON_BINDINGS_CPP})

# On Linux you may see a warning like:
#
# Warning: Unused direct dependencies:
#         /home/$USER/.local/share/ov/pkg/connectsample-200.0.0/_build/target-deps/usd/release/lib/libarch.so
#         /home/$USER/.local/share/ov/pkg/connectsample-200.0.0/_build/target-deps/usd/release/lib/libjs.so
#         /home/$USER/.local/share/ov/pkg/connectsample-200.0.0/_build/target-deps/usd/release/lib/libkind.so
#         /home/$USER/.local/share/ov/pkg/connectsample-200.0.0/_build/target-deps/usd/release/lib/libndr.so
#         /home/$USER/.local/share/ov/pkg/connectsample-200.0.0/_build/target-deps/usd/release/lib/libpcp.so
#         /home/$USER/.local/share/ov/pkg/connectsample-200.0.0/_build/target-deps/usd/release/lib/libplug.so
#         /home/$USER/.local/share/ov/pkg/connectsample-200.0.0/_build/target-deps/usd/release/lib/libsdr.so
#         /home/$USER/.local/share/ov/pkg/connectsample-200.0.0/_build/target-deps/usd/release/lib/libtrace.so
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
#     /home/slilley/.local/share/ov/pkg/connectsample-200.0.0/_build/target-deps/python/lib
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
        CesiumOmniverse
    SOURCES
        ${SOURCES}
    INCLUDE_DIRS
        "${PROJECT_SOURCE_DIR}/include"
    LIBRARIES
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
        ${CESIUM_OMNI_CXX_DEFINES}
        CESIUM_OMNI_EXPORTS
    CXX_DEFINES_DEBUG
        ${CESIUM_OMNI_CXX_DEFINES_DEBUG}
    INSTALL_COMPONENTS
        install
    INSTALL_SEARCH_PATHS
        $<TARGET_FILE_DIR:python37>
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
        ${CESIUM_OMNI_CXX_DEFINES}
    CXX_DEFINES_DEBUG
        ${CESIUM_OMNI_CXX_DEFINES_DEBUG}
    INSTALL_COMPONENTS
        kit
    INSTALL_SEARCH_PATHS
        $<TARGET_FILE_DIR:python37>
)
# cmake-format: on
