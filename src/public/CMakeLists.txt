include(Macros)

glob_files(SOURCES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

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
    LIBRARIES
        CesiumOmniverseCore
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
