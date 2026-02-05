# Utility Macros

# Set up a library
function(setup_lib)
    cmake_parse_arguments(
        ""
        ""
        "TARGET_NAME;TYPE"
        "SOURCES;INCLUDE_DIRS;PRIVATE_INCLUDE_DIRS;LIBRARIES;ADDITIONAL_LIBRARIES;DEPENDENCIES;CXX_FLAGS;CXX_FLAGS_DEBUG;CXX_DEFINES;CXX_DEFINES_DEBUG"
        ${ARGN})

    if(_TYPE)
        set(TYPE ${_TYPE})
    elseif(BUILD_SHARED_LIBS)
        set(TYPE SHARED)
    else()
        set(TYPE STATIC)
    endif()

    add_library(${_TARGET_NAME} ${TYPE})

    if(_DEPENDENCIES)
        add_dependencies(${_TARGET_NAME} ${_DEPENDENCIES})
    endif()

    add_dependencies(${_TARGET_NAME} ${_LIBRARIES})

    target_sources(${_TARGET_NAME} PRIVATE ${_SOURCES})

    target_include_directories(
        ${_TARGET_NAME}
        PUBLIC ${_INCLUDE_DIRS}
        PRIVATE ${_PRIVATE_INCLUDE_DIRS})

    target_compile_options(${_TARGET_NAME} PRIVATE ${_CXX_FLAGS} "$<$<CONFIG:DEBUG>:${_CXX_FLAGS_DEBUG}>")

    target_compile_definitions(${_TARGET_NAME} PRIVATE ${_CXX_DEFINES} "$<$<CONFIG:DEBUG>:${_CXX_DEFINES_DEBUG}>")

    # Eventually we should mark all third party libraries not in the public API as PRIVATE.
    # Note that third party libraries in the public API will need to be installed.
    target_link_libraries(${_TARGET_NAME} PUBLIC ${_LIBRARIES})

    if(CESIUM_OMNI_ENABLE_COVERAGE AND NOT WIN32)
        target_link_libraries(${_TARGET_NAME} PUBLIC gcov)
    endif()

    if(WIN32 AND ${TYPE} STREQUAL "SHARED")
        add_custom_command(
            TARGET ${_TARGET_NAME}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:${_TARGET_NAME}> $<TARGET_FILE_DIR:${_TARGET_NAME}>
            COMMAND_EXPAND_LISTS)
    endif()

    if(WIN32 AND _ADDITIONAL_LIBRARIES)
        # TARGET_RUNTIME_DLLS only works for IMPORTED targets. In some cases we can't create IMPORTED targets
        # because there's no import library (.lib) just a shared library (.dll)
        # We need to copy these to the build folder manually
        add_custom_command(
            TARGET ${_TARGET_NAME}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${_ADDITIONAL_LIBRARIES} $<TARGET_FILE_DIR:${_TARGET_NAME}>
            COMMAND_EXPAND_LISTS)
    endif()

endfunction(setup_lib)

# Set up a python module
function(setup_python_module)
    cmake_parse_arguments(
        ""
        ""
        "TARGET_NAME;PYTHON_DIR"
        "SOURCES;LIBRARIES;DEPENDENCIES;CXX_FLAGS;CXX_FLAGS_DEBUG;CXX_DEFINES;CXX_DEFINES_DEBUG"
        ${ARGN})

    add_library(${_TARGET_NAME} MODULE)

    if(_DEPENDENCIES)
        add_dependencies(${_TARGET_NAME} ${_DEPENDENCIES})
    endif()

    add_dependencies(${_TARGET_NAME} ${_LIBRARIES})

    target_sources(${_TARGET_NAME} PRIVATE ${_SOURCES})

    target_compile_options(${_TARGET_NAME} PRIVATE ${_CXX_FLAGS} "$<$<CONFIG:DEBUG>:${_CXX_FLAGS_DEBUG}>")

    target_compile_definitions(${_TARGET_NAME} PRIVATE ${_CXX_DEFINES} "$<$<CONFIG:DEBUG>:${_CXX_DEFINES_DEBUG}>")

    target_link_libraries(${_TARGET_NAME} PRIVATE ${_LIBRARIES})

    if(CESIUM_OMNI_ENABLE_COVERAGE AND NOT WIN32)
        target_link_libraries(${_TARGET_NAME} PRIVATE gcov)
    endif()

    # Pybind11 module name needs to match the file name so remove any prefix / suffix
    set_target_properties(${_TARGET_NAME} PROPERTIES DEBUG_POSTFIX "")
    set_target_properties(${_TARGET_NAME} PROPERTIES RELEASE_POSTFIX "")
    set_target_properties(${_TARGET_NAME} PROPERTIES RELWITHDEBINFO_POSTFIX "")
    set_target_properties(${_TARGET_NAME} PROPERTIES MINSIZEREL_POSTFIX "")

    if(_PYTHON_DIR)
        # Using a specific version of Python
        # Since we called find_package already in the root CMakeLists we have
        # to unset some variables. These are the variables that pybind11 cares about.
        unset(Python3_EXECUTABLE)
        unset(Python3_INTERPRETER_ID)
        unset(Python3_VERSION)
        unset(Python3_INCLUDE_DIRS)

        set(Python3_ROOT_DIR "${_PYTHON_DIR}")
        find_package(
            Python3
            COMPONENTS Interpreter
            REQUIRED)
    endif()

    # We only use pybind11 from conan for its cmake helpers
    # Note that we don't link pybind11::headers, pybind11::module, or pybind11::embed
    # The code below is a simplified version of pybind11_add_module: https://github.com/pybind/pybind11/blob/master/tools/pybind11NewTools.cmake#L174
    find_package(pybind11)

    pybind11_extension(${_TARGET_NAME})

    if(NOT DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION)
        target_link_libraries(${_TARGET_NAME} PRIVATE pybind11::lto)
    endif()

    if(MSVC)
        target_link_libraries(${_TARGET_NAME} PRIVATE pybind11::windows_extras)
    endif()

    target_link_libraries(${_TARGET_NAME} PRIVATE pybind11::opt_size)

    if(WIN32)
        add_custom_command(
            TARGET ${_TARGET_NAME}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:${_TARGET_NAME}> $<TARGET_FILE_DIR:${_TARGET_NAME}>
            COMMAND_EXPAND_LISTS)
    endif()

endfunction(setup_python_module)

# Set up a USD python library
function(setup_usd_python_lib)
    cmake_parse_arguments(
        ""
        ""
        "TARGET_NAME;PYTHON_DIR;PYTHON_MODULE_NAME;PACKAGE_NAME"
        "SOURCES;LIBRARIES;DEPENDENCIES;CXX_FLAGS;CXX_FLAGS_DEBUG;CXX_DEFINES;CXX_DEFINES_DEBUG"
        ${ARGN})

    add_library(${_TARGET_NAME} SHARED)

    if(_DEPENDENCIES)
        add_dependencies(${_TARGET_NAME} ${_DEPENDENCIES})
    endif()

    add_dependencies(${_TARGET_NAME} ${_LIBRARIES})

    target_sources(${_TARGET_NAME} PRIVATE ${_SOURCES})

    target_compile_options(${_TARGET_NAME} PRIVATE ${_CXX_FLAGS} "$<$<CONFIG:DEBUG>:${_CXX_FLAGS_DEBUG}>")

    target_compile_definitions(${_TARGET_NAME} PRIVATE ${_CXX_DEFINES} "$<$<CONFIG:DEBUG>:${_CXX_DEFINES_DEBUG}>")

    # cmake-format: off
    target_compile_definitions(${_TARGET_NAME}
        PRIVATE
        MFB_PACKAGE_NAME=${_PACKAGE_NAME}
        MFB_ALT_PACKAGE_NAME=${_PACKAGE_NAME}
        MFB_PACKAGE_MODULE=${_PYTHON_MODULE_NAME})
    # cmake-format: on

    target_link_libraries(${_TARGET_NAME} PRIVATE ${_LIBRARIES})

    if(CESIUM_OMNI_ENABLE_COVERAGE AND NOT WIN32)
        target_link_libraries(${_TARGET_NAME} PRIVATE gcov)
    endif()

    if(WIN32)
        set_target_properties(${_TARGET_NAME} PROPERTIES SUFFIX ".pyd")
    else()
        set_target_properties(${_TARGET_NAME} PROPERTIES SUFFIX ".so")
    endif()

    set_target_properties(${_TARGET_NAME} PROPERTIES PREFIX "")

    if(_PYTHON_DIR)
        # Using a specific version of Python
        # Since we called find_package already in the root CMakeLists we have
        # to unset some variables. These are the variables that pybind11 cares about.
        unset(Python3_EXECUTABLE)
        unset(Python3_INTERPRETER_ID)
        unset(Python3_VERSION)
        unset(Python3_INCLUDE_DIRS)

        set(Python3_ROOT_DIR "${_PYTHON_DIR}")
        find_package(
            Python3
            COMPONENTS Interpreter
            REQUIRED)
    endif()

    if(WIN32)
        add_custom_command(
            TARGET ${_TARGET_NAME}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:${_TARGET_NAME}> $<TARGET_FILE_DIR:${_TARGET_NAME}>
            COMMAND_EXPAND_LISTS)
    endif()

endfunction(setup_usd_python_lib)

# set up an application
function(setup_app)
    cmake_parse_arguments(
        ""
        ""
        "TARGET_NAME"
        "SOURCES;LIBRARIES;DEPENDENCIES;CXX_FLAGS;CXX_FLAGS_DEBUG;CXX_DEFINES;CXX_DEFINES_DEBUG;LINKER_FLAGS;LINKER_FLAGS_DEBUG"
        ${ARGN})

    add_executable(${_TARGET_NAME})

    if(_DEPENDENCIES)
        add_dependencies(${_TARGET_NAME} ${_DEPENDENCIES})
    endif()

    add_dependencies(${_TARGET_NAME} ${_LIBRARIES})

    target_sources(${_TARGET_NAME} PRIVATE ${_SOURCES})

    target_compile_options(${_TARGET_NAME} PRIVATE ${_CXX_FLAGS} "$<$<CONFIG:DEBUG>:${_CXX_FLAGS_DEBUG}>")

    target_compile_definitions(${_TARGET_NAME} PRIVATE ${_CXX_DEFINES} "$<$<CONFIG:DEBUG>:${_CXX_DEFINES_DEBUG}>")

    target_link_options(
        ${_TARGET_NAME}
        PRIVATE
        ${_LINKER_FLAGS}
        "$<$<CONFIG:DEBUG>:${_LINKER_FLAGS_DEBUG}>")

    target_link_libraries(${_TARGET_NAME} PRIVATE ${_LIBRARIES})

    if(WIN32)
        add_custom_command(
            TARGET ${_TARGET_NAME}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:${_TARGET_NAME}> $<TARGET_FILE_DIR:${_TARGET_NAME}>
            COMMAND_EXPAND_LISTS)
    endif()

endfunction(setup_app)

function(glob_files out_var_name regexes)
    set(files "")
    foreach(arg ${ARGV})
        list(APPEND regexes_only "${arg}")
    endforeach()
    list(POP_FRONT regexes_only)
    file(
        GLOB_RECURSE
        files
        CONFIGURE_DEPENDS
        ${regexes_only})
    set(${ARGV0}
        "${files}"
        PARENT_SCOPE)
endfunction()

function(add_external_project)
    cmake_parse_arguments(
        ""
        ""
        "PROJECT_NAME;PROJECT_EXTERN_DIRECTORY;EXPECTED_DEBUG_POSTFIX;EXPECTED_RELEASE_POSTFIX;EXPECTED_RELWITHDEBINFO_POSTFIX;EXPECTED_MINSIZEREL_POSTFIX"
        "LIBRARIES;OPTIONS"
        ${ARGN})

    include(ExternalProject) # built-in CMake include for ExternalProject_Add

    # Expands to ${_EXPECTED_<CONFIG>_POSTFIX} at configuration time (usually 'd' or '')
    set(BUILD_TIME_POSTFIX
        "\
$<$<CONFIG:Debug>:${_EXPECTED_DEBUG_POSTFIX}>\
$<$<CONFIG:Release>:${_EXPECTED_RELEASE_POSTFIX}>\
$<$<CONFIG:RelWithDebInfo>:${_EXPECTED_RELWITHDEBINFO_POSTFIX}>\
$<$<CONFIG:MinSizeRel>:${_EXPECTED_MINSIZEREL_POSTFIX}>")

    set(LIBRARY_OUTPUT_PATHS "")

    if(DEFINED _LIBRARIES)
        foreach(lib IN LISTS _LIBRARIES)
            list(
                APPEND
                LIBRARY_OUTPUT_PATHS
                "${PROJECT_BINARY_DIR}/${_PROJECT_NAME}/${CMAKE_INSTALL_LIBDIR}/$<CONFIG>/${CMAKE_STATIC_LIBRARY_PREFIX}${lib}${BUILD_TIME_POSTFIX}${CMAKE_STATIC_LIBRARY_SUFFIX}"
            )
        endforeach()
    endif()

    set(PROJECT_INCLUDE_DIR "${PROJECT_BINARY_DIR}/${_PROJECT_NAME}/${CMAKE_INSTALL_INCLUDEDIR}")

    # Force include directory to exist
    # See https://stackoverflow.com/questions/45516209/cmake-how-to-use-interface-include-directories-with-externalproject
    file(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR})

    set(EXTERN_CXX_FLAGS "")

    if(MSVC)
        # See https://learn.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warning-level-1-c4530?view=msvc-170
        set(EXTERN_CXX_FLAGS ${EXTERN_CXX_FLAGS} "/EHsc")
    endif()

    # Prepend options with -D
    list(TRANSFORM _OPTIONS PREPEND -D)

    # CMAKE_ARGS doesn't escape ; characters properly so we need to do this hack for list args like CMAKE_CONFIGURATION_TYPES
    # See https://public.kitware.com/Bug/view.php?id=16137
    # cmake-format: off
    string(REPLACE ";" "|" CMAKE_CONFIGURATION_TYPES_ALT_SEP "${CMAKE_CONFIGURATION_TYPES}")
    # cmake-format: on

    ExternalProject_Add(
        ${_PROJECT_NAME}-external
        SOURCE_DIR "${_PROJECT_EXTERN_DIRECTORY}/${_PROJECT_NAME}"
        PREFIX ${_PROJECT_NAME}
        BUILD_ALWAYS
            1 # Set this to 0 to always skip the external project build step. Be sure to reset to 1 when modifying cesium-native as it's needed there.
        LIST_SEPARATOR | # Use the alternate list separator
        CMAKE_ARGS -DBUILD_SHARED_LIBS=OFF
                   -DCMAKE_GENERATOR=${CMAKE_GENERATOR}
                   -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                   -DCMAKE_CONFIGURATION_TYPES=${CMAKE_CONFIGURATION_TYPES_ALT_SEP}
                   -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                   -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                   -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
                   -DCMAKE_INSTALL_PREFIX=${PROJECT_BINARY_DIR}/${_PROJECT_NAME}
                   -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}/$<CONFIG>
                   -DCMAKE_INSTALL_INCLUDEDIR=${CMAKE_INSTALL_INCLUDEDIR}
                   -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                   -DCMAKE_MSVC_RUNTIME_LIBRARY=${CMAKE_MSVC_RUNTIME_LIBRARY}
                   -DCMAKE_CXX_FLAGS=${EXTERN_CXX_FLAGS}
                   ${_OPTIONS}
        BUILD_BYPRODUCTS ${LIBRARY_OUTPUT_PATHS})

    if(NOT DEFINED _LIBRARIES)
        # Header only
        add_library(
            ${_PROJECT_NAME}
            INTERFACE
            IMPORTED
            GLOBAL)
        target_include_directories(${_PROJECT_NAME} INTERFACE "${PROJECT_INCLUDE_DIR}")
    else()
        foreach(lib IN LISTS _LIBRARIES)
            add_library(
                ${lib}
                STATIC
                IMPORTED
                GLOBAL)
            target_include_directories(${lib} INTERFACE "${PROJECT_INCLUDE_DIR}")
            set_target_properties(
                ${lib}
                PROPERTIES
                    IMPORTED_LOCATION_DEBUG
                    "${PROJECT_BINARY_DIR}/${_PROJECT_NAME}/${CMAKE_INSTALL_LIBDIR}/Debug/${CMAKE_STATIC_LIBRARY_PREFIX}${lib}${_EXPECTED_DEBUG_POSTFIX}${CMAKE_STATIC_LIBRARY_SUFFIX}"
                    IMPORTED_LOCATION_RELEASE
                    "${PROJECT_BINARY_DIR}/${_PROJECT_NAME}/${CMAKE_INSTALL_LIBDIR}/Release/${CMAKE_STATIC_LIBRARY_PREFIX}${lib}${_EXPECTED_RELEASE_POSTFIX}${CMAKE_STATIC_LIBRARY_SUFFIX}"
                    IMPORTED_LOCATION_RELWITHDEBINFO
                    "${PROJECT_BINARY_DIR}/${_PROJECT_NAME}/${CMAKE_INSTALL_LIBDIR}/RelWithDebInfo/${CMAKE_STATIC_LIBRARY_PREFIX}${lib}${_EXPECTED_RELWITHDEBINFO_POSTFIX}${CMAKE_STATIC_LIBRARY_SUFFIX}"
                    IMPORTED_LOCATION_MINSIZEREL
                    "${PROJECT_BINARY_DIR}/${_PROJECT_NAME}/${CMAKE_INSTALL_LIBDIR}/MinSizeRel/${CMAKE_STATIC_LIBRARY_PREFIX}${lib}${_EXPECTED_MINSIZEREL_POSTFIX}${CMAKE_STATIC_LIBRARY_SUFFIX}"
            )
        endforeach()
    endif()
endfunction()

function(add_prebuilt_project)
    cmake_parse_arguments(
        ""
        ""
        "RELEASE_INCLUDE_DIR;DEBUG_INCLUDE_DIR;RELEASE_LIBRARY_DIR;RELEASE_DLL_DIR;DEBUG_LIBRARY_DIR;DEBUG_DLL_DIR"
        "RELEASE_LIBRARIES;RELEASE_DLL_LIBRARIES;DEBUG_LIBRARIES;DEBUG_DLL_LIBRARIES;TARGET_NAMES;TARGETS_IN_DLL_DIR"
        ${ARGN})

    if(NOT DEFINED _RELEASE_DLL_LIBRARIES)
        set(_RELEASE_DLL_LIBRARIES ${_RELEASE_LIBRARIES})
    endif()

    if(NOT DEFINED _DEBUG_DLL_LIBRARIES)
        set(_DEBUG_DLL_LIBRARIES ${_DEBUG_LIBRARIES})
    endif()

    if(NOT DEFINED _RELEASE_DLL_DIR)
        set(_RELEASE_DLL_DIR ${_RELEASE_LIBRARY_DIR})
    endif()

    if(NOT DEFINED _DEBUG_DLL_DIR)
        set(_DEBUG_DLL_DIR ${_DEBUG_LIBRARY_DIR})
    endif()

    foreach(
        lib IN
        ZIP_LISTS
        _TARGET_NAMES
        _RELEASE_LIBRARIES
        _RELEASE_DLL_LIBRARIES
        _DEBUG_LIBRARIES
        _DEBUG_DLL_LIBRARIES)
        set(TARGET_NAME ${lib_0})
        set(RELEASE_NAME ${lib_1})
        set(RELEASE_DLL_NAME ${lib_2})
        set(DEBUG_NAME ${lib_3})
        set(DEBUG_DLL_NAME ${lib_4})
        add_library(
            ${TARGET_NAME}
            SHARED
            IMPORTED
            GLOBAL)
        set(TARGET_INCLUDE_DIRECTORY
            "\
$<$<CONFIG:Debug>:${_DEBUG_INCLUDE_DIR}>\
$<$<CONFIG:Release>:${_RELEASE_INCLUDE_DIR}>\
$<$<CONFIG:RelWithDebInfo>:${_RELEASE_INCLUDE_DIR}>\
$<$<CONFIG:MinSizeRel>:${_RELEASE_INCLUDE_DIR}>")

        target_include_directories(${TARGET_NAME} INTERFACE "${TARGET_INCLUDE_DIRECTORY}")

        if(WIN32)
            find_library(
                ${TARGET_NAME}_IMPLIB_RELEASE
                NAMES ${RELEASE_NAME}
                PATHS ${_RELEASE_LIBRARY_DIR}
                NO_DEFAULT_PATH NO_CACHE)
            find_library(
                ${TARGET_NAME}_IMPLIB_DEBUG
                NAMES ${DEBUG_NAME}
                PATHS ${_DEBUG_LIBRARY_DIR}
                NO_DEFAULT_PATH NO_CACHE)

            # Determine which directory to use for DLLs
            # If TARGETS_IN_DLL_DIR is empty, default to DLL_DIR for all targets
            # Otherwise, only targets in TARGETS_IN_DLL_DIR use DLL_DIR; others use LIBRARY_DIR
            if(NOT _TARGETS_IN_DLL_DIR)
                # Empty list - default all to DLL_DIR
                set(RELEASE_DLL_LOCATION "${_RELEASE_DLL_DIR}")
                set(DEBUG_DLL_LOCATION "${_DEBUG_DLL_DIR}")
            elseif(TARGET_NAME IN_LIST _TARGETS_IN_DLL_DIR)
                # Target is in the DLL_DIR list
                set(RELEASE_DLL_LOCATION "${_RELEASE_DLL_DIR}")
                set(DEBUG_DLL_LOCATION "${_DEBUG_DLL_DIR}")
            else()
                # Target is not in the DLL_DIR list - use LIBRARY_DIR
                set(RELEASE_DLL_LOCATION "${_RELEASE_LIBRARY_DIR}")
                set(DEBUG_DLL_LOCATION "${_DEBUG_LIBRARY_DIR}")
            endif()

            set(${TARGET_NAME}_LIBRARY_RELEASE "${RELEASE_DLL_LOCATION}/${RELEASE_DLL_NAME}.dll")
            set(${TARGET_NAME}_LIBRARY_DEBUG "${DEBUG_DLL_LOCATION}/${DEBUG_DLL_NAME}.dll")
        else()
            find_library(
                ${TARGET_NAME}_LIBRARY_RELEASE
                NAMES ${RELEASE_NAME}
                PATHS ${_RELEASE_LIBRARY_DIR}
                NO_DEFAULT_PATH NO_CACHE)
            find_library(
                ${TARGET_NAME}_LIBRARY_DEBUG
                NAMES ${DEBUG_NAME}
                PATHS ${_DEBUG_LIBRARY_DIR}
                NO_DEFAULT_PATH NO_CACHE)
        endif()

        mark_as_advanced(${TARGET_NAME}_LIBRARY_RELEASE ${TARGET_NAME}_IMPLIB_RELEASE)
        mark_as_advanced(${TARGET_NAME}_LIBRARY_DEBUG ${TARGET_NAME}_IMPLIB_DEBUG)

        if(WIN32)
            set_target_properties(
                ${TARGET_NAME}
                PROPERTIES IMPORTED_IMPLIB_DEBUG "${${TARGET_NAME}_IMPLIB_DEBUG}"
                           IMPORTED_IMPLIB_RELEASE "${${TARGET_NAME}_IMPLIB_RELEASE}"
                           IMPORTED_IMPLIB_RELWITHDEBINFO "${${TARGET_NAME}_IMPLIB_RELEASE}"
                           IMPORTED_IMPLIB_MINSIZEREL "${${TARGET_NAME}_IMPLIB_RELEASE}")
        endif()

        set_target_properties(
            ${TARGET_NAME}
            PROPERTIES IMPORTED_LOCATION_DEBUG "${${TARGET_NAME}_LIBRARY_DEBUG}"
                       IMPORTED_LOCATION_RELEASE "${${TARGET_NAME}_LIBRARY_RELEASE}"
                       IMPORTED_LOCATION_RELWITHDEBINFO "${${TARGET_NAME}_LIBRARY_RELEASE}"
                       IMPORTED_LOCATION_MINSIZEREL "${${TARGET_NAME}_LIBRARY_RELEASE}")
    endforeach()
endfunction()

function(add_prebuilt_project_import_library_only)
    cmake_parse_arguments(
        ""
        ""
        "RELEASE_INCLUDE_DIR;DEBUG_INCLUDE_DIR;RELEASE_LIBRARY_DIR;DEBUG_LIBRARY_DIR"
        "RELEASE_LIBRARIES;DEBUG_LIBRARIES;TARGET_NAMES"
        ${ARGN})

    foreach(
        lib IN
        ZIP_LISTS
        _TARGET_NAMES
        _RELEASE_LIBRARIES
        _DEBUG_LIBRARIES)
        set(TARGET_NAME ${lib_0})
        set(RELEASE_NAME ${lib_1})
        set(DEBUG_NAME ${lib_2})
        add_library(
            ${TARGET_NAME}
            STATIC
            IMPORTED
            GLOBAL)
        set(TARGET_INCLUDE_DIRECTORY
            "\
$<$<CONFIG:Debug>:${_DEBUG_INCLUDE_DIR}>\
$<$<CONFIG:Release>:${_RELEASE_INCLUDE_DIR}>\
$<$<CONFIG:RelWithDebInfo>:${_RELEASE_INCLUDE_DIR}>\
$<$<CONFIG:MinSizeRel>:${_RELEASE_INCLUDE_DIR}>")

        target_include_directories(${TARGET_NAME} INTERFACE "${TARGET_INCLUDE_DIRECTORY}")

        find_library(
            ${TARGET_NAME}_LIBRARY_RELEASE
            NAMES ${RELEASE_NAME}
            PATHS ${_RELEASE_LIBRARY_DIR}
            NO_DEFAULT_PATH NO_CACHE)
        find_library(
            ${TARGET_NAME}_LIBRARY_DEBUG
            NAMES ${DEBUG_NAME}
            PATHS ${_DEBUG_LIBRARY_DIR}
            NO_DEFAULT_PATH NO_CACHE)

        set_target_properties(
            ${TARGET_NAME}
            PROPERTIES IMPORTED_LOCATION_DEBUG "${${TARGET_NAME}_LIBRARY_DEBUG}"
                       IMPORTED_LOCATION_RELEASE "${${TARGET_NAME}_LIBRARY_RELEASE}"
                       IMPORTED_LOCATION_RELWITHDEBINFO "${${TARGET_NAME}_LIBRARY_RELEASE}"
                       IMPORTED_LOCATION_MINSIZEREL "${${TARGET_NAME}_LIBRARY_RELEASE}")
    endforeach()
endfunction()

function(add_prebuilt_project_header_only)
    cmake_parse_arguments(
        ""
        ""
        "INCLUDE_DIR;TARGET_NAME"
        ""
        ${ARGN})

    add_library(
        ${_TARGET_NAME}
        INTERFACE
        IMPORTED
        GLOBAL)
    target_include_directories(${_TARGET_NAME} INTERFACE "${_INCLUDE_DIR}")

endfunction()
