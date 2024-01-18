include(CompilerToolFinder)

# Additional steps to perform clang-format and clang-tidy
function(setup_linters)
    cmake_parse_arguments(
        ""
        ""
        "PROJECT_SCRIPTS_DIRECTORY;PROJECT_BUILD_DIRECTORY;ENABLE_LINTERS_ON_BUILD"
        "PROJECT_SOURCE_DIRECTORIES"
        ${ARGN})

    if(NOT _PROJECT_SCRIPTS_DIRECTORY)
        message(FATAL_ERROR "PROJECT_SCRIPTS_DIRECTORY was not specified")
    endif()

    if(NOT _PROJECT_BUILD_DIRECTORY)
        message(FATAL_ERROR "PROJECT_BUILD_DIRECTORY was not specified")
    endif()

    if(NOT DEFINED _ENABLE_LINTERS_ON_BUILD)
        message(FATAL_ERROR "ENABLE_LINTERS_ON_BUILD was not specified")
    endif()

    if(NOT _PROJECT_SOURCE_DIRECTORIES)
        message(FATAL_ERROR "PROJECT_SOURCE_DIRECTORIES was not specified")
    endif()

    # Add clang-format
    get_compiler_tool_with_correct_version(
        TOOL_NAME
        "clang-format"
        TOOLCHAIN_NAME
        "Clang"
        RESULT_TOOL_PATH
        CLANG_FORMAT_PATH)

    if(NOT CLANG_FORMAT_PATH)
        message(FATAL_ERROR "Could not find clang-format in your path.")
    endif()

    add_custom_target(
        clang-format-check-all
        COMMAND "${Python3_EXECUTABLE}" "${_PROJECT_SCRIPTS_DIRECTORY}/clang_format.py" --check --all
                --clang-format-executable "${CLANG_FORMAT_PATH}" --source-directories ${_PROJECT_SOURCE_DIRECTORIES})

    add_custom_target(
        clang-format-fix-all
        COMMAND "${Python3_EXECUTABLE}" "${_PROJECT_SCRIPTS_DIRECTORY}/clang_format.py" --fix --all
                --clang-format-executable "${CLANG_FORMAT_PATH}" --source-directories ${_PROJECT_SOURCE_DIRECTORIES})

    add_custom_target(
        clang-format-check-staged
        COMMAND "${Python3_EXECUTABLE}" "${_PROJECT_SCRIPTS_DIRECTORY}/clang_format.py" --check --staged
                --clang-format-executable "${CLANG_FORMAT_PATH}" --source-directories ${_PROJECT_SOURCE_DIRECTORIES})

    add_custom_target(
        clang-format-fix-staged
        COMMAND "${Python3_EXECUTABLE}" "${_PROJECT_SCRIPTS_DIRECTORY}/clang_format.py" --fix --staged
                --clang-format-executable "${CLANG_FORMAT_PATH}" --source-directories ${_PROJECT_SOURCE_DIRECTORIES})

    # Add clang-tidy
    # our clang-tidy options are located in `.clang-tidy` in the root folder
    # when clang-tidy runs it will look for this file
    get_compiler_tool_with_correct_version(
        TOOL_NAME
        "clang-tidy"
        TOOLCHAIN_NAME
        "Clang"
        RESULT_TOOL_PATH
        CLANG_TIDY_PATH)

    if(NOT CLANG_TIDY_PATH)
        message(FATAL_ERROR "Could not find clang-tidy in your path.")
    endif()

    # CMake has built-in support for running clang-tidy during the build
    if(_ENABLE_LINTERS_ON_BUILD)
        set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_PATH})
        set(CMAKE_CXX_CLANG_TIDY
            ${CMAKE_CXX_CLANG_TIDY}
            PARENT_SCOPE)
    endif()

    add_custom_target(
        clang-tidy-staged
        COMMAND "${Python3_EXECUTABLE}" "${_PROJECT_SCRIPTS_DIRECTORY}/clang_tidy.py" --clang-tidy-executable
                "${CLANG_TIDY_PATH}" -p "${_PROJECT_BUILD_DIRECTORY}" # path that contains a compile_commands.json
    )

    # Generate a CMake target that runs clang-tidy by itself
    # `run-clang-tidy` is a python script that comes with llvm that runs clang-tidy in parallel over a compile_commands.json
    # See: https://clang.llvm.org/extra/doxygen/run-clang-tidy_8py_source.html
    get_compiler_tool_with_correct_version(
        TOOL_NAME
        "run-clang-tidy"
        TOOLCHAIN_NAME
        "Clang"
        RESULT_TOOL_PATH
        CLANG_TIDY_RUNNER_PATH)

    set(SOURCE_EXTENSIONS
        "*.cpp"
        "*.h"
        "*.cxx"
        "*.hxx"
        "*.hpp"
        "*.cc"
        "*.inl")
    foreach(source_directory ${_PROJECT_SOURCE_DIRECTORIES})
        foreach(source_extension ${SOURCE_EXTENSIONS})
            file(GLOB_RECURSE source_directory_files ${source_directory}/${source_extension})
            list(APPEND all_source_files ${source_directory_files})
        endforeach()
    endforeach()

    if(CLANG_TIDY_RUNNER_PATH)
        add_custom_target(
            clang-tidy
            COMMAND
                ${Python3_EXECUTABLE} ${CLANG_TIDY_RUNNER_PATH} -clang-tidy-binary ${CLANG_TIDY_PATH} -p
                ${_PROJECT_BUILD_DIRECTORY} # path that contains a compile_commands.json
                ${all_source_files})
        add_custom_target(
            clang-tidy-fix
            COMMAND
                ${Python3_EXECUTABLE} ${CLANG_TIDY_RUNNER_PATH} -fix -clang-tidy-binary ${CLANG_TIDY_PATH} -p
                ${_PROJECT_BUILD_DIRECTORY} # path that contains a compile_commands.json
                ${all_source_files})
    else()
        # run-clang-tidy was not found, so call clang-tidy directly.
        # this takes a lot longer because it's not parallelized.
        add_custom_target(
            clang-tidy
            COMMAND ${CLANG_TIDY_PATH} -p ${_PROJECT_BUILD_DIRECTORY} # path that contains a compile_commands.json
                    ${all_source_files})
        add_custom_target(
            clang-tidy-fix
            COMMAND
                ${CLANG_TIDY_PATH} --fix -p ${_PROJECT_BUILD_DIRECTORY} # path that contains a compile_commands.json
                ${all_source_files})
    endif()
endfunction()
