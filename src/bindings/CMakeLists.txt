include(Macros)

glob_files(SOURCES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

set(CESIUM_OMNIVERSE_PYTHON_BINDINGS_CXX_DEFINES ${CESIUM_OMNI_CXX_DEFINES})

if(NOT BUILD_SHARED_LIBS)
    set(CESIUM_OMNIVERSE_PYTHON_BINDINGS_CXX_DEFINES ${CESIUM_OMNIVERSE_PYTHON_BINDINGS_CXX_DEFINES} CESIUM_OMNI_STATIC)
endif()

# cmake-format: off
setup_python_module(
    PYTHON_DIR
        # Use the same Python version as Omniverse (Python 3.7)
        "${PROJECT_SOURCE_DIR}/extern/nvidia/_build/target-deps/python"
    TARGET_NAME
        CesiumOmniversePythonBindings
    SOURCES
        ${SOURCES}
    LIBRARIES
        CesiumOmniverse
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
