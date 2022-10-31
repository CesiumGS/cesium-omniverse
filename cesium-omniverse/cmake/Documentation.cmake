find_package(Doxygen) # System library

function(setup_doxygen_if_available)
    if(DOXYGEN_FOUND)
        cmake_parse_arguments(
            ""
            ""
            "PROJECT_ROOT_DIRECTORY;OUTPUT_DIRECTORY"
            "PROJECT_INCLUDE_DIRECTORIES"
            ${ARGN})

        if(NOT _PROJECT_ROOT_DIRECTORY)
            message(FATAL_ERROR "PROJECT_ROOT_DIRECTORY was not specified")
        endif()

        if(NOT _OUTPUT_DIRECTORY)
            message(FATAL_ERROR "OUTPUT_DIRECTORY was not specified")
        endif()

        if(NOT _PROJECT_INCLUDE_DIRECTORIES)
            message(FATAL_ERROR "PROJECT_INCLUDE_DIRECTORIES was not specified")
        endif()

        set(DOXYGEN_OUTPUT_DIRECTORY "${_OUTPUT_DIRECTORY}")
        set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${_PROJECT_ROOT_DIRECTORY}/README.md")

        # Hide namespace and class scopes before all functions
        # See https://www.doxygen.nl/manual/config.html#cfg_hide_scope_names
        set(DOXYGEN_HIDE_SCOPE_NAMES YES)

        doxygen_add_docs(
            generate-documentation "${_PROJECT_INCLUDE_DIRECTORIES}" "${_PROJECT_ROOT_DIRECTORY}/README.md"
            WORKING_DIRECTORY "${_PROJECT_ROOT_DIRECTORY}"
            COMMENT "Generate HTML documentation")
    endif()
endfunction()
