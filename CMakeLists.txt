cmake_minimum_required(VERSION 3.22)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.24.0")
    # Tell CMake not to set CC and CXX flags. We'll do this ourselves
    # https://cmake.org/cmake/help/latest/policy/CMP0132.html
    cmake_policy(SET CMP0132 NEW)
endif()

# Both CXX and C need to be given otherwise Conan may ignore the CMAKE_C_COMPILER flag
project(
    CesiumOmniverse
    VERSION 0.24.0
    DESCRIPTION "Cesium for Omniverse"
    LANGUAGES CXX C)

# Set the CC and CXX environment variables explicitly so that conan picks them up
set(ENV{CC} ${CMAKE_C_COMPILER})
set(ENV{CXX} ${CMAKE_CXX_COMPILER})

# Ensure that all packages are built in the same .conan directory
# Windows Long Paths must be enabled. See instructions in README.
if(WIN32)
    set(ENV{CONAN_USER_HOME_SHORT} "None")
endif()

# Just in case we need to pin a revision in AddConanDependencies
set(ENV{CONAN_REVISIONS_ENABLED} "1")

# cxx version
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT Python3_EXECUTABLE)
    # Find Python for Conan and scripts
    find_package(
        Python3
        COMPONENTS Interpreter
        REQUIRED)
endif()

# Options:
option(CESIUM_OMNI_ENABLE_TESTS "Unit tests" ON)
option(CESIUM_OMNI_ENABLE_DOCUMENTATION "Generate HTML documentation with Doxygen" ON)
option(CESIUM_OMNI_ENABLE_SANITIZERS "Check for undefined behavior at runtime" OFF)
option(CESIUM_OMNI_ENABLE_LINTERS "Enable clang-format for code formatting and clang-tidy for static code analysis" ON)
option(CESIUM_OMNI_ENABLE_TRACING "Whether to enable the Cesium performance tracing framework" OFF)
option(CESIUM_OMNI_USE_NVIDIA_DEBUG_LIBRARIES "Whether to build using NVIDIA debug libraries" OFF)

if(CESIUM_OMNI_USE_NVIDIA_DEBUG_LIBRARIES)
    set(USE_NVIDIA_RELEASE_LIBRARIES FALSE)
else()
    set(USE_NVIDIA_RELEASE_LIBRARIES TRUE)
endif()

if(MSVC)
    if(${USE_NVIDIA_RELEASE_LIBRARIES})
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
        set(MSVC_RUNTIME_FLAG_DEBUG "/MD")
        set(MSVC_RUNTIME_FLAG_RELEASE "/MD")

        # Disable run-time error checks. Otherwise boost will error with the message "Using the /RTC option without specifying a debug runtime will lead to linker errors"
        # cmake-format: off
        STRING(REGEX REPLACE "/RTC1" "" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
        STRING(REGEX REPLACE "/RTC1" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
        # cmake-format: on
    else()
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
        set(MSVC_RUNTIME_FLAG_DEBUG "/MDd")
        set(MSVC_RUNTIME_FLAG_RELEASE "/MD")
    endif()

    # Explicitly set the C/CXX flags for conan
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${MSVC_RUNTIME_FLAG_DEBUG} /MP")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${MSVC_RUNTIME_FLAG_RELEASE} /MP")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} ${MSVC_RUNTIME_FLAG_RELEASE} /MP")
    set(CMAKE_C_FLAGS_MINSIZEREL "${CMAKE_C_FLAGS_MINSIZEREL} ${MSVC_RUNTIME_FLAG_RELEASE} /MP")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${MSVC_RUNTIME_FLAG_DEBUG} /MP")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${MSVC_RUNTIME_FLAG_RELEASE} /MP")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${MSVC_RUNTIME_FLAG_RELEASE} /MP")
    set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} ${MSVC_RUNTIME_FLAG_RELEASE} /MP")
endif()

# Tell CMake where to look for our CMake helper files when calling `include`
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Built-in CMake include for using `cmake_dependent_option`
include(CMakeDependentOption)

# Coverage can only be enabled if tests are also enabled
cmake_dependent_option(
    CESIUM_OMNI_ENABLE_COVERAGE
    "Code coverage for checking the thoroughness of unit tests - only available if CESIUM_OMNI_ENABLE_TESTS is ON"
    OFF
    CESIUM_OMNI_ENABLE_TESTS
    OFF)
cmake_dependent_option(
    CESIUM_OMNI_ENABLE_LINTERS_ON_BUILD
    "Check linting during the build - only available if CESIUM_OMNI_ENABLE_LINTERS is ON"
    OFF
    CESIUM_OMNI_ENABLE_LINTERS
    OFF)
cmake_dependent_option(
    CESIUM_OMNI_ENABLE_LINTERS_ON_COMMIT
    "Check formatting and linting before committing code - only available if CESIUM_OMNI_ENABLE_LINTERS is ON"
    OFF
    CESIUM_OMNI_ENABLE_LINTERS
    OFF)

# Change the default build type to Release
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE
        Release
        CACHE STRING "Type of build: Debug Release MinSizeRel RelWithDebInfo." FORCE)
endif()

# Generate compile_commands.json file for use by language servers such as clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Source directories for formatting and linting
set(LINT_SOURCE_DIRECTORIES
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/src/bindings"
    "${PROJECT_SOURCE_DIR}/src/core"
    "${PROJECT_SOURCE_DIR}/src/public"
    "${PROJECT_SOURCE_DIR}/tests")

# Source directories for coverage
set(COVERAGE_SOURCE_DIRECTORIES "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/src")

# Source directories for documentation
set(DOC_SOURCE_DIRECTORIES "${PROJECT_SOURCE_DIR}/include")

if(CESIUM_OMNI_ENABLE_DOCUMENTATION)
    include(Documentation)
    setup_doxygen_if_available(
        PROJECT_ROOT_DIRECTORY
        "${PROJECT_SOURCE_DIR}"
        PROJECT_INCLUDE_DIRECTORIES
        ${DOC_SOURCE_DIRECTORIES}
        OUTPUT_DIRECTORY
        "${PROJECT_BINARY_DIR}/docs")
endif()

set(SCRIPTS_DIRECTORY "${PROJECT_SOURCE_DIR}/scripts")

if(CESIUM_OMNI_ENABLE_LINTERS)
    include(Linters)
    setup_linters(
        PROJECT_SCRIPTS_DIRECTORY
        "${SCRIPTS_DIRECTORY}"
        PROJECT_BUILD_DIRECTORY
        "${PROJECT_BINARY_DIR}"
        PROJECT_SOURCE_DIRECTORIES
        ${LINT_SOURCE_DIRECTORIES}
        ENABLE_LINTERS_ON_BUILD
        ${CESIUM_OMNI_ENABLE_LINTERS_ON_BUILD})
endif()

include(InstallHooks)
if(CESIUM_OMNI_ENABLE_LINTERS_ON_COMMIT)
    install_git_hooks(
        PROJECT_ROOT_DIRECTORY
        "${PROJECT_SOURCE_DIR}"
        GIT_HOOKS_SOURCE_DIRECTORY
        "${SCRIPTS_DIRECTORY}")
else()
    uninstall_git_hooks(PROJECT_ROOT_DIRECTORY "${PROJECT_SOURCE_DIR}")
endif()

set(CMAKE_LINK_WHAT_YOU_USE ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

# defines CMAKE_INSTALL_LIBDIR, CMAKE_INSTALL_BINDIR, and CMAKE_INSTALL_INCLUDEDIR for OS-dependent build locations.
# See https://cmake.org/cmake/help/latest/module/GNUInstallDirs.html
include(GNUInstallDirs)

# Add conan third party libraries
include(AddConanDependencies)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")

# compiler / preprocessor / linker options passed later to targets in cmake/Macros.cmake
# Debug options are only applied if using the Debug configuration
set(CESIUM_OMNI_CXX_FLAGS "")
set(CESIUM_OMNI_CXX_FLAGS_DEBUG "")
set(CESIUM_OMNI_CXX_DEFINES "")
set(CESIUM_OMNI_CXX_DEFINES_DEBUG "")
set(CESIUM_OMNI_LINKER_FLAGS "")
set(CESIUM_OMNI_LINKER_FLAGS_DEBUG "")

if(CESIUM_OMNI_ENABLE_TRACING)
    set(CESIUM_OMNI_CXX_DEFINES ${CESIUM_OMNI_CXX_DEFINES} "CESIUM_TRACING_ENABLED")
endif()

set(CESIUM_OMNI_CXX_DEFINES ${CESIUM_OMNI_CXX_DEFINES} CESIUM_OMNI_VERSION="${PROJECT_VERSION}")
set(CESIUM_OMNI_CXX_DEFINES ${CESIUM_OMNI_CXX_DEFINES} CESIUM_DISABLE_DEFAULT_ELLIPSOID)

# Add a define CESIUM_OMNI_GIT_HASH_ABBREVIATED
execute_process(
    COMMAND git describe --match=NeVeRmAtCh --always --abbrev=7
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    OUTPUT_VARIABLE CESIUM_OMNI_GIT_HASH_ABBREVIATED
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

set(CESIUM_OMNI_CXX_DEFINES ${CESIUM_OMNI_CXX_DEFINES}
                            CESIUM_OMNI_GIT_HASH_ABBREVIATED="${CESIUM_OMNI_GIT_HASH_ABBREVIATED}")

# Get the git tag
execute_process(
    COMMAND git describe --tags --exact-match --abbrev=0
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    OUTPUT_VARIABLE CESIUM_OMNI_GIT_TAG
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

# Add DEBUG definition to be able to #ifdef
set(CESIUM_OMNI_CXX_DEFINES_DEBUG ${CESIUM_OMNI_CXX_DEFINES_DEBUG} CESIUM_OMNI_DEBUG)

if(MSVC)
    set(CESIUM_OMNI_CXX_FLAGS
        ${CESIUM_OMNI_CXX_FLAGS}
        /W4 # enable all warnings
        /WX # error on warnings
        /Zc:__cplusplus # make __cplusplus defined correctly
        # /Zc:preprocessor
        /bigobj
    )
    set(CESIUM_OMNI_CXX_DEFINES
        ${CESIUM_OMNI_CXX_DEFINES}
        _CRT_SECURE_NO_WARNINGS # Avoid erroring on strcpy
        UNICODE # tell Windows to use the unicode version of string functions, as opposed to ASCII (https://docs.microsoft.com/en-us/windows/win32/learnwin32/working-with-strings)
        _UNICODE # tell Windows to use the unicode version of string functions, as opposed to ASCII (https://docs.microsoft.com/en-us/windows/win32/learnwin32/working-with-strings)
        WIN32_LEAN_AND_MEAN # ignore some unnecessary Windows APIs
        NOMINMAX # don't let Windows create macros for MIN and MAX
        _SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING
        _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR # See https://developercommunity.visualstudio.com/t/Visual-Studio-17100-Update-leads-to-Pr/10669759?sort=newest
    )
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set(CESIUM_OMNI_CXX_FLAGS
        ${CESIUM_OMNI_CXX_FLAGS}
        -Werror
        -Wall
        -Wextra
        -Wpedantic
        -Wparentheses
        -Wreturn-type
        -Wmissing-braces
        -Wunknown-pragmas
        -Wshadow
        -Wunused
        -Wuninitialized
        -Wconversion
        -Wno-missing-field-initializers
        -Wno-return-local-addr
        -Wno-array-bounds # Ignore false positives in libstdc++ in GCC 13.1.1
        -fstrict-aliasing
        -fstrict-enums)

    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        set(CESIUM_OMNI_CXX_FLAGS ${CESIUM_OMNI_CXX_FLAGS} -Wno-stringop-overflow) # Ignore false positives in libstdc++ in GCC 13.1.1
    endif()

    if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
        set(CESIUM_OMNI_CXX_FLAGS ${CESIUM_OMNI_CXX_FLAGS} -Wno-gnu-anonymous-struct)
    endif()
endif()

if(CESIUM_OMNI_ENABLE_SANITIZERS)
    if(MSVC)
        set(SANITIZER_FLAGS /fsanitize=address /fno-sanitize-address-vcasan-lib)
        set(CESIUM_OMNI_CXX_FLAGS ${CESIUM_OMNI_CXX_FLAGS} ${SANITIZER_FLAGS})
    else()
        set(SANITIZER_FLAGS
            -fsanitize=address
            -fsanitize=undefined
            -fsanitize=float-divide-by-zero
            -fsanitize=bounds
            -fno-omit-frame-pointer)

        if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
            # These flags are only supported in clang
            # Can't use -fsanitize=unsigned-integer-overflow since it triggers UB in libstdc++ std::map: https://github.com/google/sanitizers/issues/1346
            set(SANITIZER_FLAGS
                ${SANITIZER_FLAGS}
                -fsanitize=implicit-conversion
                -fsanitize=local-bounds
                -fsanitize=nullability)
        endif()

        set(CESIUM_OMNI_CXX_FLAGS ${CESIUM_OMNI_CXX_FLAGS} ${SANITIZER_FLAGS})
        set(CESIUM_OMNI_LINKER_FLAGS ${CESIUM_OMNI_LINKER_FLAGS} ${SANITIZER_FLAGS})
    endif()
endif()

# Enable code coverage if supported on this platform.
if(CESIUM_OMNI_ENABLE_COVERAGE)
    # At reconfigure time: Delete any lingering GCDA files in the build folder.
    # GCDA files track profiling information and are generated everytime a binary with
    # coverage support is executed. If we don't do this we can encounter a
    # `cannot merge previous GCDA file: corrupt arc tag` error message (when switching
    # compilers).
    file(GLOB_RECURSE GCDA_FILES "${PROJECT_BINARY_DIR}/*.gcda")
    foreach(GCDA IN LISTS GCDA_FILES)
        file(REMOVE "${GCDA}")
    endforeach()

    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
        set(CESIUM_OMNI_CXX_FLAGS_DEBUG ${CESIUM_OMNI_CXX_FLAGS_DEBUG} -fprofile-arcs -ftest-coverage)
        set(CESIUM_OMNI_LINKER_FLAGS_DEBUG ${CESIUM_OMNI_LINKER_FLAGS_DEBUG} --coverage)
    endif()

    # COVERAGE_SOURCE_DIRECTORIES must be passed exactly like this
    # If you:
    # - Don't escape the semicolon list delimiters
    # - Don't use the VERBATIM option with add_custom_target
    # Then CMake will swap your semicolons with whitespaces, ruining
    # the list when it's forwarded to the generate-coverage script and
    # causing only the first directory in your list to actually be used
    # for filters.

    # cmake-format: off
    list(JOIN COVERAGE_SOURCE_DIRECTORIES "\;" COVERAGE_SOURCE_DIRECTORIES_FORMATTED)
    # cmake-format: on

    # We always define the `generate-coverage` target, even if it would fail at runtime
    # due to missing tools / incompatible compilers.
    add_custom_target(
        generate-coverage
        COMMAND
            ${CMAKE_COMMAND} -DCMAKE_CXX_COMPILER_ID=${CMAKE_CXX_COMPILER_ID}
            -DCMAKE_CXX_COMPILER_VERSION=${CMAKE_CXX_COMPILER_VERSION} -DPROJECT_ROOT_DIRECTORY=${PROJECT_SOURCE_DIR}
            -DPROJECT_BUILD_DIRECTORY=${PROJECT_BINARY_DIR}
            -DPROJECT_SOURCE_DIRECTORIES=${COVERAGE_SOURCE_DIRECTORIES_FORMATTED}
            -DOUTPUT_DIRECTORY=${PROJECT_BINARY_DIR}/coverage -P ${PROJECT_SOURCE_DIR}/cmake/GenerateCoverage.cmake
        DEPENDS $<TARGET_FILE:cesium.omniverse.cpp.tests.plugin>
        VERBATIM)
endif()

# System preprocessor definitions
# CESIUM_OMNI_WINDOWS = The program is being compiled for Windows
# CESIUM_OMNI_MSVC = The program is being compiled for Windows using the Microsoft C++ compiler
if(WIN32)
    set(CESIUM_OMNI_CXX_DEFINES ${CESIUM_OMNI_CXX_DEFINES} CESIUM_OMNI_WINDOWS)
    if(MSVC)
        set(CESIUM_OMNI_CXX_DEFINES ${CESIUM_OMNI_CXX_DEFINES} CESIUM_OMNI_MSVC)
    endif()
elseif(APPLE)
    set(CESIUM_OMNI_CXX_DEFINES ${CESIUM_OMNI_CXX_DEFINES} CESIUM_OMNI_MACOS)
elseif(UNIX)
    set(CESIUM_OMNI_CXX_DEFINES ${CESIUM_OMNI_CXX_DEFINES} CESIUM_OMNI_UNIX)
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CESIUM_OMNI_CXX_DEFINES ${CESIUM_OMNI_CXX_DEFINES} CESIUM_OMNI_GCC)
endif()

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set(CESIUM_OMNI_CXX_DEFINES ${CESIUM_OMNI_CXX_DEFINES} CESIUM_OMNI_CLANG)
endif()

# Enable/Disable sanitizers
if(CESIUM_OMNI_ENABLE_SANITIZERS)
    set(CESIUM_OMNI_CXX_DEFINES ${CESIUM_OMNI_CXX_DEFINES} CESIUM_OMNI_ENABLE_SANITIZERS)
endif()

set(CESIUM_OMNI_CXX_DEFINES
    ${CESIUM_OMNI_CXX_DEFINES}
    GLM_FORCE_XYZW_ONLY # Disable .rgba and .stpq to make it easier to view values from debugger
    GLM_FORCE_EXPLICIT_CTOR # Disallow implicit conversions between dvec3 <-> dvec4, dvec3 <-> fvec3, etc
    GLM_FORCE_SIZE_T_LENGTH # Make vec.length() and vec[idx] use size_t instead of int
    GLM_ENABLE_EXPERIMENTAL
)

# Boost is a dependency of USD. It is dynamically linked so we need to set this flag.
set(CESIUM_OMNI_CXX_DEFINES ${CESIUM_OMNI_CXX_DEFINES} BOOST_ALL_DYN_LINK)

set(CESIUM_OMNI_CXX_DEFINES ${CESIUM_OMNI_CXX_DEFINES} BOOST_ALL_NO_LIB)

# Hide USD build warning
set(CESIUM_OMNI_CXX_DEFINES ${CESIUM_OMNI_CXX_DEFINES} BOOST_BIND_GLOBAL_PLACEHOLDERS)

# TBB is a dependency of USD
if(NOT ${USE_NVIDIA_RELEASE_LIBRARIES})
    set(CESIUM_OMNI_CXX_DEFINES_DEBUG ${CESIUM_OMNI_CXX_DEFINES_DEBUG} TBB_USE_DEBUG)
endif()

# Add this for cesium-native (TODO: should this go in src/CMakeLists?)
set(CESIUM_OMNI_CXX_DEFINES ${CESIUM_OMNI_CXX_DEFINES} SPDLOG_COMPILED_LIB)

# Add this for cesium-native (TODO: should this go in src/CMakeLists?)
set(CESIUM_OMNI_CXX_DEFINES ${CESIUM_OMNI_CXX_DEFINES} LIBASYNC_STATIC)

# Needed to include USD headers without errors
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CESIUM_OMNI_CXX_DEFINES ${CESIUM_OMNI_CXX_DEFINES} _GLIBCXX_PERMIT_BACKWARD_HASH)
endif()


set(CESIUM_OMNI_CXX_FLAGS
    ${CESIUM_OMNI_CXX_FLAGS}
    CACHE STRING "Compiler flags to use." FORCE)
set(CESIUM_OMNI_CXX_FLAGS_DEBUG
    ${CESIUM_OMNI_CXX_FLAGS_DEBUG}
    CACHE STRING "Compiler flags to use in debug mode." FORCE)
set(CESIUM_OMNI_CXX_DEFINES
    ${CESIUM_OMNI_CXX_DEFINES}
    CACHE STRING "Preprocessor definitions to use." FORCE)
set(CESIUM_OMNI_CXX_DEFINES_DEBUG
    ${CESIUM_OMNI_CXX_DEFINES_DEBUG}
    CACHE STRING "Preprocessor definitions to use in debug mode." FORCE)
set(CESIUM_OMNI_LINKER_FLAGS
    ${CESIUM_OMNI_LINKER_FLAGS}
    CACHE STRING "Linker flags to use." FORCE)
set(CESIUM_OMNI_LINKER_FLAGS_DEBUG
    ${CESIUM_OMNI_LINKER_FLAGS_DEBUG}
    CACHE STRING "Linker flags to use in debug mode." FORCE)

# vcpkg setup
set(VCPKG_OVERLAY_PORTS "${CMAKE_CURRENT_LIST_DIR}/vcpkg/ports")
set(VCPKG_OVERLAY_TRIPLETS "${CMAKE_CURRENT_LIST_DIR}/vcpkg/triplets")

include(${CMAKE_CURRENT_SOURCE_DIR}/extern/cesium-native/cmake/detect-vcpkg-triplet.cmake)

if (NOT VCPKG_TRIPLET)
    if (DEFINED ENV{VCPKG_TRIPLET})
        set(VCPKG_TRIPLET "$ENV{VCPKG_TRIPLET}")
    elseif(DETECTED_VCPKG_TRIPLET_ERROR)
        message(FATAL_ERROR "${DETECTED_VCPKG_TRIPLET_ERROR}")
    else()
        set(VCPKG_TRIPLET "${DETECTED_VCPKG_TRIPLET}-omniverse")
    endif()
endif()

if (NOT EZVCPKG_BASEDIR)
    if (DEFINED ENV{EZVCPKG_BASEDIR})
        set(EZVCPKG_BASEDIR "$ENV{EZVCPKG_BASEDIR}")
    endif()
endif()

# Third party dependencies not handled by conan
add_subdirectory(extern)

# Tell CMake to search for packages with the correct build type (i.e. Release or Debug) as generated
# by ConfigureConan.cmake with the cmake_find_package_multi generator
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG TRUE)

find_package(Threads) # System threading library

# Conan libraries
find_package(CURL)
find_package(doctest)
find_package(stb)
find_package(ZLIB)
find_package(yaml-cpp)

# So that the installed libraries can find shared libraries in the same directory
set(CMAKE_INSTALL_RPATH $ORIGIN)

# Cesium Omniverse library
add_subdirectory(src)

# unit tests
if(CESIUM_OMNI_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Ninja and various Makefiles generators support generating compile_commands.json
# https://cmake.org/cmake/help/latest/variable/CMAKE_EXPORT_COMPILE_COMMANDS.html
# https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html#makefile-generators
if("${CMAKE_GENERATOR}" MATCHES "Ninja"
   OR "${CMAKE_GENERATOR}" MATCHES "Makefiles"
   OR "${CMAKE_GENERATOR}" MATCHES "WMake")
    # Copy compile_commands.json to the root directory after configuration so that
    # extensions like clangd or intellisense use the correct compile flags
    add_custom_target(
        copy-compile-commands ALL
        ${CMAKE_COMMAND}
        -E
        copy_if_different
        "${PROJECT_BINARY_DIR}/compile_commands.json"
        "${PROJECT_SOURCE_DIR}")
endif()

# Generate ThirdParty.json
execute_process(COMMAND "${Python3_EXECUTABLE}" "${SCRIPTS_DIRECTORY}/generate_third_party_license_json.py" --build-dir
                        "${PROJECT_BINARY_DIR}" --project-dir "${PROJECT_SOURCE_DIR}" --skip "doctest,strawberryperl")

# Copy docs and related resources to exts folder.
execute_process(COMMAND "${Python3_EXECUTABLE}" "${SCRIPTS_DIRECTORY}/copy_to_exts.py")

# When calling find_package with `CMAKE_FIND_PACKAGE_PREFER_CONFIG` it adds
# a `<PACKAGE>_DIR` variable to the CMake cache, polluting the CMake GUI.
# Use `mark_as_advanced` on all packages to hide them from the GUI.
# Conan internally does `mark_as_advanced` for `cmake_find_package` generator, but not for
# `cmake_find_package_multi`. Related: https://github.com/conan-io/conan/issues/6167
get_cmake_property(PACKAGE_NAMES PACKAGES_FOUND)
foreach(PACKAGE_NAME ${PACKAGE_NAMES})
    mark_as_advanced(${PACKAGE_NAME}_DIR)
endforeach()

# Hide CONAN_CMD_PATH from CMake GUI
mark_as_advanced(CONAN_CMD_PATH)

# Configure plugInfo.json for CesiumUsdSchemas
set(PLUG_INFO_RESOURCE_PATH "resources")
set(PLUG_INFO_ROOT "..")

set(PLUG_INFO_LIBRARY_FILE_NAME "${CMAKE_SHARED_LIBRARY_PREFIX}CesiumUsdSchemas${CMAKE_SHARED_LIBRARY_SUFFIX}")

set(PLUG_INFO_LIBRARY_PATH_BUILD "../../${PLUG_INFO_LIBRARY_FILE_NAME}")
set(PLUG_INFO_LIBRARY_PATH_INSTALL "../../lib/${PLUG_INFO_LIBRARY_FILE_NAME}")
set(PLUG_INFO_LIBRARY_PATH_KIT "../../bin/${PLUG_INFO_LIBRARY_FILE_NAME}")

set(PLUG_INFO_LIBRARY_PATH ${PLUG_INFO_LIBRARY_PATH_BUILD})
configure_file("${PROJECT_SOURCE_DIR}/src/plugins/CesiumUsdSchemas/plugInfo.json.in"
               "${PROJECT_BINARY_DIR}/CesiumUsdSchemas.plugInfo.json.build")
set(PLUG_INFO_LIBRARY_PATH ${PLUG_INFO_LIBRARY_PATH_INSTALL})
configure_file("${PROJECT_SOURCE_DIR}/src/plugins/CesiumUsdSchemas/plugInfo.json.in"
               "${PROJECT_BINARY_DIR}/CesiumUsdSchemas.plugInfo.json.install")
set(PLUG_INFO_LIBRARY_PATH ${PLUG_INFO_LIBRARY_PATH_KIT})
configure_file("${PROJECT_SOURCE_DIR}/src/plugins/CesiumUsdSchemas/plugInfo.json.in"
               "${PROJECT_BINARY_DIR}/CesiumUsdSchemas.plugInfo.json.kit")

# Configure generatedSchema.usda for CesiumUsdSchemas
configure_file("${PROJECT_SOURCE_DIR}/src/plugins/CesiumUsdSchemas/generatedSchema.usda.in"
               "${PROJECT_BINARY_DIR}/CesiumUsdSchemas.generatedSchema.usda.build")
configure_file("${PROJECT_SOURCE_DIR}/src/plugins/CesiumUsdSchemas/generatedSchema.usda.in"
               "${PROJECT_BINARY_DIR}/CesiumUsdSchemas.generatedSchema.usda.install")
configure_file("${PROJECT_SOURCE_DIR}/src/plugins/CesiumUsdSchemas/generatedSchema.usda.in"
               "${PROJECT_BINARY_DIR}/CesiumUsdSchemas.generatedSchema.usda.kit")

set(KIT_EXTENSION_BIN_PATH "${PROJECT_SOURCE_DIR}/exts/cesium.omniverse/bin")
set(KIT_EXTENSION_CERTS_PATH "${PROJECT_SOURCE_DIR}/exts/cesium.omniverse/certs/cacert.pem")
set(KIT_PLUGIN_EXTENSION_BIN_PATH "${PROJECT_SOURCE_DIR}/exts/cesium.usd.plugins/bin")
set(KIT_EXTENSION_BINDINGS_PATH "${PROJECT_SOURCE_DIR}/exts/cesium.omniverse/cesium/omniverse/bindings")
set(KIT_PLUGIN_EXTENSION_BINDINGS_PATH "${PROJECT_SOURCE_DIR}/exts/cesium.usd.plugins/cesium/usd/plugins")
set(KIT_EXTENSION_PLUGINS_PATH "${PROJECT_SOURCE_DIR}/exts/cesium.usd.plugins/plugins")

set(KIT_EXTENSION_TESTS_BIN_PATH "${PROJECT_SOURCE_DIR}/exts/cesium.omniverse.cpp.tests/bin")
set(KIT_EXTENSION_TESTS_BINDINGS_PATH
    "${PROJECT_SOURCE_DIR}/exts/cesium.omniverse.cpp.tests/cesium/omniverse/cpp/tests/bindings")

# You may see warnings like:
#
# CMake Warning at build/src/cmake_install.cmake:115 (file):
#   Dependency libpython3.10.so.1.0 found in search directory:
#     cesium-omniverse/extern/nvidia/_build/target-deps/python/lib
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

# Automatically fetch certificates used for any requests from the core Context class
execute_process(COMMAND "${Python3_EXECUTABLE}" "${SCRIPTS_DIRECTORY}/update_certs.py" "${KIT_EXTENSION_CERTS_PATH}")

get_property(INSTALL_SEARCH_PATHS GLOBAL PROPERTY NVIDIA_ADDITIONAL_SEARCH_PATHS_PROPERTY)

# Skip system libraries when running install
set(INSTALL_PRE_EXCLUDE_REGEXES "api-ms-*" "ext-ms-*")
set(INSTALL_POST_EXCLUDE_REGEXES "system32" "^\/lib")

install(
    TARGETS CesiumUsdSchemas
    ARCHIVE DESTINATION ${KIT_PLUGIN_EXTENSION_BIN_PATH} COMPONENT install
    LIBRARY DESTINATION ${KIT_PLUGIN_EXTENSION_BIN_PATH} COMPONENT install
    RUNTIME DESTINATION ${KIT_PLUGIN_EXTENSION_BIN_PATH} COMPONENT install)

install(
    TARGETS CesiumUsdSchemas
            RUNTIME_DEPENDENCIES
            DIRECTORIES
            ${INSTALL_SEARCH_PATHS}
            PRE_EXCLUDE_REGEXES
            ${INSTALL_PRE_EXCLUDE_REGEXES}
            POST_EXCLUDE_REGEXES
            ${INSTALL_POST_EXCLUDE_REGEXES}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT library
            EXCLUDE_FROM_ALL
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT library
            EXCLUDE_FROM_ALL
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT library
            EXCLUDE_FROM_ALL)

install(
    TARGETS cesium.omniverse.plugin
            RUNTIME_DEPENDENCIES
            PRE_INCLUDE_REGEXES
            ${INSTALL_PRE_INCLUDE_REGEXES}
            PRE_EXCLUDE_REGEXES
            ".*"
            POST_EXCLUDE_REGEXES
            ${INSTALL_POST_EXCLUDE_REGEXES}
    ARCHIVE DESTINATION ${KIT_EXTENSION_BIN_PATH} COMPONENT install
    LIBRARY DESTINATION ${KIT_EXTENSION_BIN_PATH} COMPONENT install
    RUNTIME DESTINATION ${KIT_EXTENSION_BIN_PATH} COMPONENT install)

# TODO: this is also installing the .lib. How to avoid that?
install(
    TARGETS cesium.omniverse.plugin
            RUNTIME_DEPENDENCIES
            DIRECTORIES
            ${INSTALL_SEARCH_PATHS}
            PRE_EXCLUDE_REGEXES
            ${INSTALL_PRE_EXCLUDE_REGEXES}
            POST_EXCLUDE_REGEXES
            ${INSTALL_POST_EXCLUDE_REGEXES}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT library
            EXCLUDE_FROM_ALL
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT library
            EXCLUDE_FROM_ALL
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT library
            EXCLUDE_FROM_ALL)

install(
    TARGETS CesiumOmniversePythonBindings
    ARCHIVE DESTINATION ${KIT_EXTENSION_BINDINGS_PATH} COMPONENT install
    LIBRARY DESTINATION ${KIT_EXTENSION_BINDINGS_PATH} COMPONENT install
    RUNTIME DESTINATION ${KIT_EXTENSION_BINDINGS_PATH} COMPONENT install)

install(
    TARGETS CesiumOmniversePythonBindings
            RUNTIME_DEPENDENCIES
            DIRECTORIES
            ${INSTALL_SEARCH_PATHS}
            PRE_EXCLUDE_REGEXES
            ${INSTALL_PRE_EXCLUDE_REGEXES}
            POST_EXCLUDE_REGEXES
            ${INSTALL_POST_EXCLUDE_REGEXES}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT library
            EXCLUDE_FROM_ALL
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT library
            EXCLUDE_FROM_ALL
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT library
            EXCLUDE_FROM_ALL)

install(
    TARGETS _CesiumUsdSchemas
    ARCHIVE DESTINATION "${KIT_PLUGIN_EXTENSION_BINDINGS_PATH}/CesiumUsdSchemas" COMPONENT install
    LIBRARY DESTINATION "${KIT_PLUGIN_EXTENSION_BINDINGS_PATH}/CesiumUsdSchemas" COMPONENT install
    RUNTIME DESTINATION "${KIT_PLUGIN_EXTENSION_BINDINGS_PATH}/CesiumUsdSchemas" COMPONENT install)

install(
    TARGETS _CesiumUsdSchemas
            RUNTIME_DEPENDENCIES
            DIRECTORIES
            ${INSTALL_SEARCH_PATHS}
            PRE_EXCLUDE_REGEXES
            ${INSTALL_PRE_EXCLUDE_REGEXES}
            POST_EXCLUDE_REGEXES
            ${INSTALL_POST_EXCLUDE_REGEXES}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT library
            EXCLUDE_FROM_ALL
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT library
            EXCLUDE_FROM_ALL
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT library
            EXCLUDE_FROM_ALL)

if(CESIUM_OMNI_ENABLE_TESTS)
    install(
        TARGETS cesium.omniverse.cpp.tests.plugin
        ARCHIVE DESTINATION ${KIT_EXTENSION_TESTS_BIN_PATH} COMPONENT install
        LIBRARY DESTINATION ${KIT_EXTENSION_TESTS_BIN_PATH} COMPONENT install
        RUNTIME DESTINATION ${KIT_EXTENSION_TESTS_BIN_PATH} COMPONENT install)
    install(
        TARGETS cesium.omniverse.cpp.tests.plugin
                RUNTIME_DEPENDENCIES
                DIRECTORIES
                ${INSTALL_SEARCH_PATHS}
                PRE_EXCLUDE_REGEXES
                ${INSTALL_PRE_EXCLUDE_REGEXES}
                POST_EXCLUDE_REGEXES
                ${INSTALL_POST_EXCLUDE_REGEXES}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                COMPONENT library
                EXCLUDE_FROM_ALL
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
                COMPONENT library
                EXCLUDE_FROM_ALL
        RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}
                COMPONENT library
                EXCLUDE_FROM_ALL)

    install(
        TARGETS CesiumOmniverseCppTestsPythonBindings
        ARCHIVE DESTINATION ${KIT_EXTENSION_TESTS_BINDINGS_PATH} COMPONENT install
        LIBRARY DESTINATION ${KIT_EXTENSION_TESTS_BINDINGS_PATH} COMPONENT install
        RUNTIME DESTINATION ${KIT_EXTENSION_TESTS_BINDINGS_PATH} COMPONENT install)

    install(
        TARGETS CesiumOmniverseCppTestsPythonBindings
                RUNTIME_DEPENDENCIES
                DIRECTORIES
                ${INSTALL_SEARCH_PATHS}
                PRE_EXCLUDE_REGEXES
                ${INSTALL_PRE_EXCLUDE_REGEXES}
                POST_EXCLUDE_REGEXES
                ${INSTALL_POST_EXCLUDE_REGEXES}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                COMPONENT library
                EXCLUDE_FROM_ALL
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
                COMPONENT library
                EXCLUDE_FROM_ALL
        RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}
                COMPONENT library
                EXCLUDE_FROM_ALL)
endif()

install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT library
    EXCLUDE_FROM_ALL)

install(
    FILES "${PROJECT_BINARY_DIR}/CesiumUsdSchemas.plugInfo.json.kit"
    DESTINATION "${KIT_EXTENSION_PLUGINS_PATH}/CesiumUsdSchemas/resources/"
    RENAME "plugInfo.json"
    COMPONENT install)

install(
    FILES "${PROJECT_BINARY_DIR}/CesiumUsdSchemas.plugInfo.json.install"
    DESTINATION "plugins/CesiumUsdSchemas/resources/"
    RENAME "plugInfo.json"
    COMPONENT library
    EXCLUDE_FROM_ALL)

install(
    FILES "${PROJECT_BINARY_DIR}/CesiumUsdSchemas.generatedSchema.usda.kit"
    DESTINATION "${KIT_EXTENSION_PLUGINS_PATH}/CesiumUsdSchemas/resources/"
    RENAME "generatedSchema.usda"
    COMPONENT install)

install(
    FILES "${PROJECT_BINARY_DIR}/CesiumUsdSchemas.generatedSchema.usda.install"
    DESTINATION "plugins/CesiumUsdSchemas/resources/"
    RENAME "generatedSchema.usda"
    COMPONENT library
    EXCLUDE_FROM_ALL)

install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/exts/cesium.omniverse"
    DESTINATION .
    COMPONENT package
    EXCLUDE_FROM_ALL
    PATTERN "__pycache__" EXCLUDE)

install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/exts/cesium.usd.plugins"
    DESTINATION .
    COMPONENT package
    EXCLUDE_FROM_ALL
    PATTERN "__pycache__" EXCLUDE)

set(CPACK_GENERATOR "ZIP")
set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
set(CPACK_COMPONENTS_ALL package)
set(CPACK_COMPONENTS_ALL_IN_ONE_PACKAGE ON)
set(CPACK_PACKAGE_NAME CesiumGS-cesium-omniverse)

string(TOLOWER ${CMAKE_SYSTEM_NAME} SYSTEM_NAME_LOWERCASE)
set(SYSTEM_PROCESSOR "x86_64")

if(CESIUM_OMNI_GIT_TAG)
    set(CPACK_PACKAGE_FILE_NAME
        ${CPACK_PACKAGE_NAME}-${SYSTEM_NAME_LOWERCASE}-${SYSTEM_PROCESSOR}-${CESIUM_OMNI_GIT_TAG})
else()
    set(CPACK_PACKAGE_FILE_NAME
        ${CPACK_PACKAGE_NAME}-${SYSTEM_NAME_LOWERCASE}-${SYSTEM_PROCESSOR}-${CESIUM_OMNI_GIT_HASH_ABBREVIATED})
endif()

include(CPack)
