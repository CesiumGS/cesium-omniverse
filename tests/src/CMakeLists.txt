include(Macros)

glob_files(SOURCES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

get_property(ADDITIONAL_LIBRARIES GLOBAL PROPERTY NVIDIA_ADDITIONAL_LIBRARIES_PROPERTY)

# cmake-format: off
setup_lib(
    TARGET_NAME
        cesium.omniverse.cpp.tests.plugin
    TYPE
        # Carbonite Plugins needs to be shared libraries
        SHARED
    SOURCES
        ${SOURCES}
    INCLUDE_DIRS
        "${PROJECT_SOURCE_DIR}/tests/include"
    LIBRARIES
        CesiumOmniverseCore
        doctest::doctest
        yaml-cpp::yaml-cpp
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
