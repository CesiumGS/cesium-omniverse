# cmake-conan configuration

macro(configure_conan)
    cmake_parse_arguments(
        ""
        ""
        "PROJECT_BUILD_DIRECTORY"
        "REQUIRES;OPTIONS"
        ${ARGN})

    if(NOT _PROJECT_BUILD_DIRECTORY)
        message(FATAL_ERROR "PROJECT_BUILD_DIRECTORY was not specified")
    endif()

    if(NOT _REQUIRES)
        message(FATAL_ERROR "REQUIRES was not specified")
    endif()

    if(NOT EXISTS "${_PROJECT_BUILD_DIRECTORY}/conan.cmake")
        message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
        file(
            DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/0.17.0/conan.cmake"
            "${_PROJECT_BUILD_DIRECTORY}/conan.cmake"
            EXPECTED_HASH SHA256=3bef79da16c2e031dc429e1dac87a08b9226418b300ce004cc125a82687baeef
            TLS_VERIFY ON)
    endif()

    include("${_PROJECT_BUILD_DIRECTORY}/conan.cmake")

    # Execute conan config init to ensure ~/.conan/settings.yml exists
    find_program(CONAN_CMD_PATH conan)
    if(NOT CONAN_CMD_PATH)
        message(FATAL_ERROR "Could not find conan in system path!")
    endif()
    execute_process(COMMAND "${CONAN_CMD_PATH}" config init)

    # Find the location of the .conan directory.
    execute_process(
        COMMAND "${CONAN_CMD_PATH}" config home
        OUTPUT_VARIABLE CONAN_DIRECTORY
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    # Conan doesn't support marking a dependency as "always build if not cached" directly.
    # You're allowed to mark a dependency as "always build" (which ignores the cache on rebuilds) or build if "missing".
    # Build if missing means that Conan will only build the package if and only if your local settings are incompatible
    # with the binary version of the package at https://conan.io/center/
    #
    # You can force incompatible settings by using custom settings, but these settings have to be present in your ~/.conan/settings.yml
    # or Conan will throw an error about an unrecognized option. This is an (ugly) workaround that injects a os.distro="BuildFromSource"
    # option in that file so that we can trigger package incompatibility intentionally and avoid downloading binary packages.
    #
    # See the following links for details:
    # https://docs.conan.io/en/latest/extending/custom_settings.html?highlight=glibc#adding-new-settings
    # https://github.com/conan-io/conan/issues/2749
    # https://github.com/conan-io/conan/issues/1692
    # https://github.com/conan-io/conan/issues/7117
    if(WIN32)
        set(CONAN_OS_NAME "Windows")
    elseif(APPLE)
        set(CONAN_OS_NAME "Macos")
    elseif(UNIX)
        set(CONAN_OS_NAME "Linux")
    endif()

    set(CONAN_SETTINGS_FILE "${CONAN_DIRECTORY}/settings.yml")
    set(ADD_DISTRO "${CONAN_OS_NAME}:\n        distro: [None, \"BuildFromSource\"]")
    file(READ "${CONAN_SETTINGS_FILE}" settings_text)
    string(FIND ${settings_text} "${ADD_DISTRO}" position)
    if(position EQUAL "-1")
        string(
            REPLACE "${CONAN_OS_NAME}:"
                    ${ADD_DISTRO}
                    new_settings_text
                    ${settings_text})
        file(WRITE "${CONAN_SETTINGS_FILE}" "${new_settings_text}")
    endif()

    # Add "revisions_enabled = 1" to ~/.conan/conan.conf
    # This is needed in order for the Dependency Graph build target to work
    set(CONAN_CONF_FILE "${CONAN_DIRECTORY}/conan.conf")
    set(ADD_REVISIONS_ENABLED "[general]\nrevisions_enabled = 1")
    file(READ "${CONAN_CONF_FILE}" conf_text)
    string(FIND ${conf_text} "${ADD_REVISIONS_ENABLED}" position)
    if(position EQUAL "-1")
        string(
            REPLACE "[general]"
                    ${ADD_REVISIONS_ENABLED}
                    new_conf_text
                    ${conf_text})
        file(WRITE "${CONAN_CONF_FILE}" "${new_conf_text}")
    endif()

    conan_cmake_configure(
        REQUIRES
        ${_REQUIRES}
        OPTIONS
        ${_OPTIONS}
        GENERATORS
        cmake_find_package_multi)

    # CMAKE_CONFIGURATION_TYPES will not be defined with single configuration
    # generators, so populate with CMAKE_BUILD_TYPE as a fallback.
    # CMAKE_BUILD type is always defined in the root CMakeLists.txt
    set(CONAN_CONFIGURATION_TYPES "${CMAKE_CONFIGURATION_TYPES}")
    if(NOT CONAN_CONFIGURATION_TYPES)
        set(CONAN_CONFIGURATION_TYPES ${CMAKE_BUILD_TYPE})
    endif()

    set(CONAN_PACKAGE_INSTALL_FOLDER "${_PROJECT_BUILD_DIRECTORY}/Conan_Packages")
    list(APPEND CMAKE_MODULE_PATH ${CONAN_PACKAGE_INSTALL_FOLDER})
    list(APPEND CMAKE_PREFIX_PATH ${CONAN_PACKAGE_INSTALL_FOLDER})

    foreach(type ${CONAN_CONFIGURATION_TYPES})
        conan_cmake_autodetect(settings BUILD_TYPE ${type})
        list(APPEND settings "os.distro=BuildFromSource")
        if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
            # Build with old C++ ABI. See top-level CMakeLists.txt for explanation.
            # See https://docs.conan.io/en/latest/howtos/manage_gcc_abi.html
            list(APPEND settings "compiler.libcxx=libstdc++")
        endif()
        conan_cmake_install(
            PATH_OR_REFERENCE
            ${_PROJECT_BUILD_DIRECTORY}
            BUILD
            missing
            INSTALL_FOLDER
            ${CONAN_PACKAGE_INSTALL_FOLDER}
            REMOTE
            conancenter
            SETTINGS
            ${settings}
            UPDATE)
    endforeach()

    # Hide CONAN_CMD from the CMake GUI
    mark_as_advanced(CONAN_CMD)

endmacro()
