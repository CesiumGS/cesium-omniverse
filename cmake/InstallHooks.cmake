function(install_git_hooks)
    cmake_parse_arguments(
        ""
        ""
        "PROJECT_ROOT_DIRECTORY;GIT_HOOKS_SOURCE_DIRECTORY"
        ""
        ${ARGN})

    if(NOT _PROJECT_ROOT_DIRECTORY)
        message(FATAL_ERROR "PROJECT_ROOT_DIRECTORY was not specified")
    endif()

    if(NOT _GIT_HOOKS_SOURCE_DIRECTORY)
        message(FATAL_ERROR "GIT_HOOKS_SOURCE_DIRECTORY was not specified")
    endif()

    get_git_directory(
        PROJECT_ROOT_DIRECTORY
        ${_PROJECT_ROOT_DIRECTORY}
        RESULT_GIT_HOOKS_DIRECTORY
        GIT_HOOKS_DESTINATION_DIRECTORY)

    file(COPY "${_GIT_HOOKS_SOURCE_DIRECTORY}/pre-commit" DESTINATION "${GIT_HOOKS_DESTINATION_DIRECTORY}/")
    file(MAKE_DIRECTORY "${GIT_HOOKS_DESTINATION_DIRECTORY}/utils/")
    file(COPY "${_GIT_HOOKS_SOURCE_DIRECTORY}/utils/" DESTINATION "${GIT_HOOKS_DESTINATION_DIRECTORY}/utils")
endfunction()

function(uninstall_git_hooks)
    cmake_parse_arguments(
        ""
        ""
        "PROJECT_ROOT_DIRECTORY;"
        ""
        ${ARGN})

    if(NOT _PROJECT_ROOT_DIRECTORY)
        message(FATAL_ERROR "PROJECT_ROOT_DIRECTORY was not specified")
    endif()

    get_git_directory(
        PROJECT_ROOT_DIRECTORY
        ${_PROJECT_ROOT_DIRECTORY}
        RESULT_GIT_HOOKS_DIRECTORY
        GIT_HOOKS_DIRECTORY)

    file(REMOVE "${GIT_HOOKS_DIRECTORY}/pre-commit")
    file(REMOVE_RECURSE "${GIT_HOOKS_DIRECTORY}/utils/")
endfunction()

function(get_git_directory)
    cmake_parse_arguments(
        ""
        ""
        "PROJECT_ROOT_DIRECTORY;RESULT_GIT_HOOKS_DIRECTORY"
        ""
        ${ARGN})

    if(NOT _PROJECT_ROOT_DIRECTORY)
        message(FATAL_ERROR "PROJECT_ROOT_DIRECTORY was not specified")
    endif()

    if(NOT _RESULT_GIT_HOOKS_DIRECTORY)
        message(FATAL_ERROR "RESULT_GIT_HOOKS_DIRECTORY was not specified")
    endif()

    # Use git rev-parse --git-common-dir to support installing the hooks even if we're in a git worktree
    set(GIT_DIR_CMD git rev-parse --git-common-dir)
    execute_process(
        COMMAND ${GIT_DIR_CMD}
        WORKING_DIRECTORY "${_PROJECT_ROOT_DIRECTORY}"
        ERROR_VARIABLE GIT_DIR_CMD_ERROR
        OUTPUT_VARIABLE GIT_DIRECTORY
        RESULT_VARIABLE GIT_DIR_CMD_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    # cmake-format: off
    if(GIT_DIR_CMD_RESULT AND NOT GIT_DIR_CMD_RESULT EQUAL 0)
        message(FATAL_ERROR "Command ${GIT_DIR_CMD} in directory ${_PROJECT_ROOT_DIRECTORY} returned non-zero result: ${GIT_DIR_CMD_RESULT}\nCommand output:${GIT_DIRECTORY}\nError: ${GIT_DIR_CMD_ERROR}")
    endif()
    # cmake-format: on

    if(NOT IS_ABSOLUTE ${GIT_DIRECTORY})
        set(GIT_DIRECTORY "${_PROJECT_ROOT_DIRECTORY}/${GIT_DIRECTORY}")
    endif()

    set(GIT_HOOKS_DESTINATION_DIRECTORY "${GIT_DIRECTORY}/hooks")

    # Normalize paths so they don't end with a trailing slash.
    cmake_path(
        NATIVE_PATH
        GIT_HOOKS_DESTINATION_DIRECTORY
        NORMALIZE
        GIT_HOOKS_DESTINATION_DIRECTORY_NORMALIZED)

    set(${_RESULT_GIT_HOOKS_DIRECTORY}
        "${GIT_HOOKS_DESTINATION_DIRECTORY_NORMALIZED}"
        PARENT_SCOPE)
endfunction()
