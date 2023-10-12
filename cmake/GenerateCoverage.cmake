# Usage: ${CMAKE_COMMAND}
# -DCMAKE_CXX_COMPILER_ID=${CMAKE_CXX_COMPILER_ID} # Required, must be "Clang", "AppleClang", or "GNU"
# -DCMAKE_CXX_COMPILER_VERSION=${CMAKE_CXX_COMPILER_VERSION} # Required. Used to auto select the gcov or llvm-cov version to use if possible.
# -DPROJECT_ROOT_DIRECTORY=${PROJECT_SOURCE_DIR} # Required, should be abs path to root project folder
# -DPROJECT_BUILD_DIRECTORY=${PROJECT_BINARY_DIR} # Required, abs path to build directory
# -DPROJECT_SOURCE_DIRECTORIES=${PROJECT_SOURCE_DIR}/src;${PROJECT_SOURCE_DIR}/include # Required, abs paths to source directories
# -DOUTPUT_DIRECTORY=${PROJECT_BINARY_DIR}/coverage # Required, directory to store generated HTML report in.
# -P cmake/GenerateCoverage.cmake

# This CMake script will:
# 1. Determine if coverage generation is supported on your platform
# 2. Delete the old GCDA files from previous test executions.
# 3. Execute the tests (it shouldn't matter if any tests fail or not, coverage is still recorded).
# 4. Executes gcovr with the necessary flags to generate an HTML report in build/coverage/index.html
# This script should always be ran at BUILD TIME, using the `generate-coverage` target.

# Tell CMake where to look for our CMake helper files when calling `include`
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")

include(CompilerToolFinder)

# Detect if dev tried to do include(GenerateCoverage) and abort.
if(NOT CMAKE_SCRIPT_MODE_FILE)
    message(FATAL_ERROR "This is a build-time CMake script that must be ran in script (-P) mode.")
endif()

if(NOT CMAKE_CXX_COMPILER_ID)
    message(FATAL_ERROR "CMAKE_CXX_COMPILER_ID must be defined")
endif()

if(NOT CMAKE_CXX_COMPILER_VERSION)
    message(FATAL_ERROR "CMAKE_CXX_COMPILER_VERSION must be defined")
endif()

if(NOT PROJECT_ROOT_DIRECTORY)
    message(FATAL_ERROR "PROJECT_ROOT_DIRECTORY must be defined")
endif()

if(NOT PROJECT_BUILD_DIRECTORY)
    message(FATAL_ERROR "PROJECT_BUILD_DIRECTORY must be defined")
endif()

if(NOT PROJECT_SOURCE_DIRECTORIES)
    message(FATAL_ERROR "PROJECT_SOURCE_DIRECTORIES must be defined")
endif()

if(NOT OUTPUT_DIRECTORY)
    message(FATAL_ERROR "OUTPUT_DIRECTORY must be defined")
endif()

find_program(GCOVR_EXECUTABLE "gcovr")
if(NOT GCOVR_EXECUTABLE)
    message(FATAL_ERROR "Could not find gcovr. Cannot generate coverage.")
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    get_compiler_tool_with_correct_version(
        TOOL_NAME
        "gcov"
        TOOLCHAIN_NAME
        "GNU"
        RESULT_TOOL_PATH
        GCOV_PATH)

    if(NOT GCOV_PATH)
        message(FATAL_ERROR "Could not find gcov in your path. Can't generate coverage with g++.")
    endif()

    set(GCOV_EXECUTABLE ${GCOV_PATH})
elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    get_compiler_tool_with_correct_version(
        TOOL_NAME
        "llvm-cov"
        TOOLCHAIN_NAME
        "Clang"
        RESULT_TOOL_PATH
        LLVM_COV_PATH)

    if(NOT LLVM_COV_PATH)
        message(FATAL_ERROR "Could not find llvm-cov in your path. Can't generate coverage with Clang.")
    endif()

    set(GCOV_EXECUTABLE ${LLVM_COV_PATH}\ gcov)
else()
    message(FATAL_ERROR "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}, cannot generate coverage.")
endif()

file(GLOB_RECURSE GCDA_FILES "${PROJECT_BUILD_DIRECTORY}/*.gcda")
list(LENGTH GCDA_FILES GCDA_FILES_LENGTH)

# if(GCDA_FILES_LENGTH GREATER 0)
#     message("Removing old GCDA files: ")
#     foreach(GCDA IN LISTS GCDA_FILES)
#         message("${GCDA}")
#         file(REMOVE "${GCDA}")
#     endforeach()
# endif()

# execute_process(COMMAND ctest WORKING_DIRECTORY "${PROJECT_BUILD_DIRECTORY}")

message("Removing and recreating ${PROJECT_BUILD_DIRECTORY}/coverage")

file(REMOVE_RECURSE "${OUTPUT_DIRECTORY}")
file(MAKE_DIRECTORY "${OUTPUT_DIRECTORY}")

set(CMD
    ${GCOVR_EXECUTABLE}
    --gcov-executable=${GCOV_EXECUTABLE}
    --delete # delete GCDA files after processing
    --txt # print text output
    --html-details # generate HTML report with annotated source code
    "${OUTPUT_DIRECTORY}/index.html")

foreach(DIR IN LISTS PROJECT_SOURCE_DIRECTORIES)
    set(CMD ${CMD} "--filter=${DIR}")
endforeach()

message("Generating HTML coverage with command: ${CMD}")

execute_process(COMMAND ${CMD} WORKING_DIRECTORY "${PROJECT_ROOT_DIRECTORY}")
