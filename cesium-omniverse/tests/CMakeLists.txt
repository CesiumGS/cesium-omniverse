include(Macros)

glob_files(SOURCES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

# cmake-format: off
setup_app(
    TARGET_NAME
        tests
    SOURCES
        ${SOURCES}
    LIBRARIES
        CesiumOmniverse
        doctest::doctest
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
