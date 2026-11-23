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

include(Macros)

glob_files(SOURCES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

get_property(ADDITIONAL_LIBRARIES GLOBAL PROPERTY NVIDIA_ADDITIONAL_LIBRARIES_PROPERTY)
get_property(ADDITIONAL_LINK_DIRECTORIES GLOBAL PROPERTY NVIDIA_ADDITIONAL_LINK_DIRECTORIES_PROPERTY)

# cmake-format: off
setup_lib(
    TARGET_NAME
        cesium.omniverse.plugin
    TYPE
        # Carbonite Plugins needs to be shared libraries
        SHARED
    SOURCES
        ${SOURCES}
    INCLUDE_DIRS
        "${PROJECT_SOURCE_DIR}/include"
        # Stub headers for rtx_plugins (not publicly available)
        "${CMAKE_CURRENT_LIST_DIR}/include/stubs"
        "${CMAKE_CURRENT_LIST_DIR}/include"
    LIBRARIES
        CesiumOmniverseCore
        fabric
        carb
        omni_kit
        omni_ui
        boost_preprocessor
        gsl
    ADDITIONAL_LIBRARIES
        # Unfortunately we need this in both cesium.omniverse.plugin and CesiumOmniverseCore because we're bypassing
        # CMake's built-in dependency system
        "${ADDITIONAL_LIBRARIES}"
    CXX_FLAGS
        ${CESIUM_OMNI_CXX_FLAGS}
    CXX_FLAGS_DEBUG
        ${CESIUM_OMNI_CXX_FLAGS_DEBUG}
    CXX_DEFINES
        ${CESIUM_OMNI_CXX_DEFINES}
    CXX_DEFINES_DEBUG
        ${CESIUM_OMNI_CXX_DEFINES_DEBUG}
)
# cmake-format: on
