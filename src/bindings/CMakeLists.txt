include(Macros)

# Use the same Python version as Omniverse (Python 3.7)
# Since we called find_package already in the root CMakeLists we have
# to unset some variables. These are the variables that pybind11 cares about.
unset(Python3_EXECUTABLE)
unset(Python3_INTERPRETER_ID)
unset(Python3_VERSION)
unset(Python3_INCLUDE_DIRS)

set(Python3_ROOT_DIR "${PROJECT_SOURCE_DIR}/extern/nvidia/_build/target-deps/python")
find_package(
    Python3
    COMPONENTS Interpreter Development
    REQUIRED)

find_package(pybind11)

glob_files(SOURCES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

set(CESIUM_OMNIVERSE_PYTHON_BINDINGS_CXX_DEFINES ${CESIUM_OMNI_CXX_DEFINES})

if(NOT BUILD_SHARED_LIBS)
    set(CESIUM_OMNIVERSE_PYTHON_BINDINGS_CXX_DEFINES ${CESIUM_OMNIVERSE_PYTHON_BINDINGS_CXX_DEFINES} CESIUM_OMNI_STATIC)
endif()

# cmake-format: off
setup_python_module(
    TARGET_NAME
        CesiumOmniversePythonBindings
    SOURCES
        ${SOURCES}
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
