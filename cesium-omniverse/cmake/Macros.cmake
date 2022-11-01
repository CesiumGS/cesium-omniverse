# Utility Macros

# Set up a library
function(setup_lib)
    cmake_parse_arguments(
        ""
        ""
        "TARGET_NAME"
        "SOURCES;INCLUDE_DIRS;PRIVATE_INCLUDE_DIRS;LIBRARIES;DEPENDENCIES;CXX_FLAGS;CXX_FLAGS_DEBUG;CXX_DEFINES;CXX_DEFINES_DEBUG;INSTALL_COMPONENTS;INSTALL_SEARCH_PATHS"
        ${ARGN})

    add_library(${_TARGET_NAME})

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

    if(WIN32 AND BUILD_SHARED_LIBS)
        add_custom_command(
            TARGET ${_TARGET_NAME}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:${_TARGET_NAME}> $<TARGET_FILE_DIR:${_TARGET_NAME}>
            COMMAND_EXPAND_LISTS)
    endif()

    foreach(COMPONENT IN LISTS _INSTALL_COMPONENTS)
        install(
            TARGETS ${_TARGET_NAME}
                    RUNTIME_DEPENDENCIES
                    DIRECTORIES
                    ${_INSTALL_SEARCH_PATHS}
                    PRE_EXCLUDE_REGEXES
                    "api-ms-*"
                    "ext-ms-*"
                    POST_EXCLUDE_REGEXES
                    "system32"
                    "^\/lib"
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT ${COMPONENT}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT ${COMPONENT}
            RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT ${COMPONENT})

        install(
            DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            COMPONENT ${COMPONENT})
    endforeach()

endfunction(setup_lib)

# Set up a python module
function(setup_python_module)
    cmake_parse_arguments(
        ""
        ""
        "TARGET_NAME"
        "SOURCES;LIBRARIES;DEPENDENCIES;CXX_FLAGS;CXX_FLAGS_DEBUG;CXX_DEFINES;CXX_DEFINES_DEBUG;INSTALL_COMPONENTS;INSTALL_SEARCH_PATHS"
        ${ARGN})

    pybind11_add_module(${_TARGET_NAME} MODULE)

    if(_DEPENDENCIES)
        add_dependencies(${_TARGET_NAME} ${_DEPENDENCIES})
    endif()

    add_dependencies(${_TARGET_NAME} ${_LIBRARIES})

    target_sources(${_TARGET_NAME} PRIVATE ${_SOURCES})

    target_compile_options(${_TARGET_NAME} PRIVATE ${_CXX_FLAGS} "$<$<CONFIG:DEBUG>:${_CXX_FLAGS_DEBUG}>")

    target_compile_definitions(${_TARGET_NAME} PRIVATE ${_CXX_DEFINES} "$<$<CONFIG:DEBUG>:${_CXX_DEFINES_DEBUG}>")

    target_link_libraries(${_TARGET_NAME} PRIVATE ${_LIBRARIES})

    # Pybind11 module name needs to match the file name so remove any prefix / suffix
    set_target_properties(${_TARGET_NAME} PROPERTIES DEBUG_POSTFIX "")
    set_target_properties(${_TARGET_NAME} PROPERTIES RELEASE_POSTFIX "")
    set_target_properties(${_TARGET_NAME} PROPERTIES RELWITHDEBINFO_POSTFIX "")
    set_target_properties(${_TARGET_NAME} PROPERTIES MINSIZEREL_POSTFIX "")

    if(WIN32)
        add_custom_command(
            TARGET ${_TARGET_NAME}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:${_TARGET_NAME}> $<TARGET_FILE_DIR:${_TARGET_NAME}>
            COMMAND_EXPAND_LISTS)
    endif()

    foreach(COMPONENT IN LISTS _INSTALL_COMPONENTS)
        install(
            TARGETS ${_TARGET_NAME}
                    # TODO: for some reason this isn't installing libCesiumOmniverse.so or its dependencies on Linux
                    RUNTIME_DEPENDENCIES
                    DIRECTORIES
                    ${_INSTALL_SEARCH_PATHS}
                    PRE_EXCLUDE_REGEXES
                    "api-ms-*"
                    "ext-ms-*"
                    POST_EXCLUDE_REGEXES
                    "system32"
                    "^\/lib"
            ARCHIVE DESTINATION .
                    COMPONENT ${COMPONENT}
                    EXCLUDE_FROM_ALL
            LIBRARY DESTINATION .
                    COMPONENT ${COMPONENT}
                    EXCLUDE_FROM_ALL
            RUNTIME DESTINATION .
                    COMPONENT ${COMPONENT}
                    EXCLUDE_FROM_ALL)
    endforeach()

endfunction(setup_python_module)

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
        "PROJECT_NAME;PROJECT_EXTERN_DIRECTORY"
        "LIBRARIES;OPTIONS"
        ${ARGN})

    include(ExternalProject) # built-in CMake include for ExternalProject_Add

    # Expands to ${CMAKE_DEBUG_POSTFIX} at configuration time (usually 'd' or '')
    set(BUILD_TIME_DEBUG_POSTFIX
        "\
$<$<CONFIG:Debug>:${CMAKE_DEBUG_POSTFIX}>\
$<$<CONFIG:Release>:${CMAKE_RELEASE_POSTFIX}>\
$<$<CONFIG:RelWithDebInfo>:${CMAKE_RELWITHDEBINFO_POSTFIX}>\
$<$<CONFIG:MinSizeRel>:${CMAKE_MINSIZEREL_POSTFIX}>")

    set(LIBRARY_OUTPUT_PATHS "")

    if(DEFINED _LIBRARIES)
        foreach(lib IN LISTS _LIBRARIES)
            list(
                APPEND
                LIBRARY_OUTPUT_PATHS
                "${PROJECT_BINARY_DIR}/${_PROJECT_NAME}/${CMAKE_INSTALL_LIBDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${lib}${BUILD_TIME_DEBUG_POSTFIX}${CMAKE_STATIC_LIBRARY_SUFFIX}"
            )
        endforeach()
    endif()

    set(PROJECT_INCLUDE_DIR "${PROJECT_BINARY_DIR}/${_PROJECT_NAME}/${CMAKE_INSTALL_INCLUDEDIR}")

    # Force include directory to exist
    # See https://stackoverflow.com/questions/45516209/cmake-how-to-use-interface-include-directories-with-externalproject
    file(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR})

    set(EXTERN_CXX_FLAGS "")

    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        # Build with old C++ ABI. See top-level CMakeLists.txt for explanation.
        set(EXTERN_CXX_FLAGS ${EXTERN_CXX_FLAGS} "-D_GLIBCXX_USE_CXX11_ABI=0")
    endif()

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
                   -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                   -DCMAKE_CONFIGURATION_TYPES=${CMAKE_CONFIGURATION_TYPES_ALT_SEP}
                   -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                   -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                   -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
                   -DCMAKE_INSTALL_PREFIX=${PROJECT_BINARY_DIR}/${_PROJECT_NAME}
                   -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
                   -DCMAKE_INSTALL_INCLUDEDIR=${CMAKE_INSTALL_INCLUDEDIR}
                   -DCMAKE_DEBUG_POSTFIX=${CMAKE_DEBUG_POSTFIX}
                   -DCMAKE_RELWITHDEBINFO_POSTFIX=${CMAKE_RELWITHDEBINFO_POSTFIX}
                   -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                   -DCMAKE_CXX_FLAGS=${EXTERN_CXX_FLAGS}
                   ${_OPTIONS}
        BUILD_BYPRODUCTS ${LIBRARY_OUTPUT_PATHS})

    if(NOT DEFINED _LIBRARIES)
        # Header only
        add_library(${_PROJECT_NAME} INTERFACE)
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
                    "${PROJECT_BINARY_DIR}/${_PROJECT_NAME}/${CMAKE_INSTALL_LIBDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${lib}${CMAKE_DEBUG_POSTFIX}${CMAKE_STATIC_LIBRARY_SUFFIX}"
                    IMPORTED_LOCATION_RELEASE
                    "${PROJECT_BINARY_DIR}/${_PROJECT_NAME}/${CMAKE_INSTALL_LIBDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${lib}${CMAKE_RELEASE_POSTFIX}${CMAKE_STATIC_LIBRARY_SUFFIX}"
                    IMPORTED_LOCATION_RELWITHDEBINFO
                    "${PROJECT_BINARY_DIR}/${_PROJECT_NAME}/${CMAKE_INSTALL_LIBDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${lib}${CMAKE_RELWITHDEBINFO_POSTFIX}${CMAKE_STATIC_LIBRARY_SUFFIX}"
                    IMPORTED_LOCATION_MINSIZEREL
                    "${PROJECT_BINARY_DIR}/${_PROJECT_NAME}/${CMAKE_INSTALL_LIBDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${lib}${CMAKE_MINSIZEREL_POSTFIX}${CMAKE_STATIC_LIBRARY_SUFFIX}"
            )
        endforeach()
    endif()
endfunction()

function(add_prebuilt_project)
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

            # cmake-format: off
            string(REGEX REPLACE "[.]lib$" ".dll" ${TARGET_NAME}_LIBRARY_RELEASE ${${TARGET_NAME}_IMPLIB_RELEASE})
            string(REGEX REPLACE "[.]lib$" ".dll" ${TARGET_NAME}_LIBRARY_DEBUG ${${TARGET_NAME}_IMPLIB_DEBUG})
            # cmake-format: on
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
