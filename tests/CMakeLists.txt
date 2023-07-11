include(Macros)

glob_files(SOURCES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

# replace a string in the utils header with the intended working dir for the test executable
set(TEST_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
configure_file("${CMAKE_CURRENT_LIST_DIR}/testUtils.h.in" "${CMAKE_CURRENT_LIST_DIR}/testUtils.h")

# cmake-format: off
setup_app(
    TARGET_NAME
        tests
    SOURCES
        ${SOURCES}
    LIBRARIES
        CesiumOmniverseCore
        doctest::doctest
        yaml-cpp::yaml-cpp
    CXX_FLAGS
        ${CESIUM_OMNI_CXX_FLAGS}
    CXX_FLAGS_DEBUG
        ${CESIUM_OMNI_CXX_FLAGS_DEBUG}
    CXX_DEFINES
        ${CESIUM_OMNI_CXX_DEFINES}
    CXX_DEFINES_DEBUG
        ${CESIUM_OMNI_CXX_DEFINES_DEBUG}
    LINKER_FLAGS
        ${CESIUM_OMNI_LINKER_FLAGS}
    LINKER_FLAGS_DEBUG
        ${CESIUM_OMNI_LINKER_FLAGS_DEBUG}
)
# cmake-format: on

# Discover tests for CTest (a.k.a make test)
doctest_discover_tests(tests ADD_LABELS 1)
