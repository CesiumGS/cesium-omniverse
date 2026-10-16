include(Macros)

set(CESIUM_ENABLE_CLANG_TIDY OFF)
set(CESIUM_INSTALL_STATIC_LIBS OFF)
set(CESIUM_INSTALL_HEADERS OFF)
set(CESIUM_TESTS_ENABLED OFF)
add_subdirectory(cesium-native EXCLUDE_FROM_ALL)

if(NOT ${USE_NVIDIA_RELEASE_LIBRARIES})
    execute_process(COMMAND "${Python3_EXECUTABLE}" "${SCRIPTS_DIRECTORY}/copy_from_dir.py" "*.user.xml"
                            "${PROJECT_SOURCE_DIR}/extern/nvidia/debug-deps" "${PROJECT_SOURCE_DIR}/extern/nvidia/deps")
endif()

if(WIN32)
    set(NVIDIA_PLATFORM_NAME "windows-x86_64")
    set(CARB_PLATFORM_NAME "windows-x86_64")
elseif(UNIX AND NOT APPLE)
    set(NVIDIA_PLATFORM_NAME "manylinux_2_35_x86_64")
    set(CARB_PLATFORM_NAME "linux-x86_64")
else()
    message(FATAL_ERROR "Only Windows and Linux are supported")
endif()

if(UNIX)
    execute_process(COMMAND bash -c "${PROJECT_SOURCE_DIR}/extern/nvidia/build.sh --platform ${NVIDIA_PLATFORM_NAME}"
                    RESULT_VARIABLE exit_code)
elseif(WIN32)
    execute_process(COMMAND cmd /C "${PROJECT_SOURCE_DIR}/extern/nvidia/build.bat --platform ${NVIDIA_PLATFORM_NAME}"
                    RESULT_VARIABLE exit_code)
endif()

# cmake-format: off
if(exit_code AND NOT exit_code EQUAL 0)
    message(FATAL_ERROR "Gathering Nvidia libraries failed")
endif()
# cmake-format: on

set(NVIDIA_RELEASE_FOLDER_NAME "release")

if(${USE_NVIDIA_RELEASE_LIBRARIES})
    set(NVIDIA_DEBUG_FOLDER_NAME "release")
else()
    set(NVIDIA_DEBUG_FOLDER_NAME "debug")
endif()

set(NVIDIA_BUILD_DIR "${PROJECT_SOURCE_DIR}/extern/nvidia/_build")
set(NVIDIA_USD_ROOT "${NVIDIA_BUILD_DIR}/target-deps/usd")
set(PYTHON_ROOT "${NVIDIA_BUILD_DIR}/target-deps/python")
set(CARB_ROOT "${NVIDIA_BUILD_DIR}/target-deps/carb_sdk_plugins")
set(KIT_SDK_ROOT "${NVIDIA_BUILD_DIR}/target-deps/kit-sdk")
set(KIT_OMNI_UI_ROOT "${NVIDIA_BUILD_DIR}/target-deps/omni-ui")
set(KIT_IMGUI_ROOT "${NVIDIA_BUILD_DIR}/target-deps/omni-kit-renderer-imgui")
set(PYBIND11_ROOT "${NVIDIA_BUILD_DIR}/target-deps/pybind11")
set(BOOST_PREPROCESSOR_ROOT "${NVIDIA_BUILD_DIR}/target-deps/boost_preprocessor")
set(GSL_ROOT "${NVIDIA_BUILD_DIR}/target-deps/gsl/include")

set(NVIDIA_USD_LIBRARIES
    usd_ar
    usd_arch
    usd_boost
    usd_gf
    usd_js
    usd_kind
    usd_pcp
    usd_plug
    usd_python
    usd_sdf
    usd_sdr
    usd_tf
    usd_trace
    usd_ts
    usd_usd
    usd_usdGeom
    usd_usdLux
    usd_usdShade
    usd_usdUI
    usd_usdUtils
    usd_usdVol
    usd_vt
    usd_work)

# Add base USD libraries
set(NVIDIA_USD_TARGET_NAMES ${NVIDIA_USD_LIBRARIES})
set(NVIDIA_USD_RELEASE_LIBRARIES ${NVIDIA_USD_LIBRARIES})
set(NVIDIA_USD_DEBUG_LIBRARIES ${NVIDIA_USD_LIBRARIES})

# Add TBB
set(NVIDIA_USD_TARGET_NAMES ${NVIDIA_USD_TARGET_NAMES} tbb)

if(WIN32)
    # Kit 110 ships oneTBB as tbb12.lib / tbb12.dll (tbb.dll does not exist).
    # The CMake target is still named `tbb`; only the underlying lib/DLL is versioned.
    set(NVIDIA_USD_RELEASE_LIBRARIES ${NVIDIA_USD_RELEASE_LIBRARIES} tbb12)
    set(NVIDIA_USD_DEBUG_LIBRARIES ${NVIDIA_USD_DEBUG_LIBRARIES} tbb12)
else()
    set(NVIDIA_USD_RELEASE_LIBRARIES ${NVIDIA_USD_RELEASE_LIBRARIES} tbb)
    set(NVIDIA_USD_DEBUG_LIBRARIES ${NVIDIA_USD_DEBUG_LIBRARIES} tbb)
endif()

if(${USE_NVIDIA_RELEASE_LIBRARIES})
    set(NVIDIA_USD_DEBUG_LIBRARIES ${NVIDIA_USD_RELEASE_LIBRARIES})
endif()

# cmake-format: off
add_prebuilt_project(
    RELEASE_INCLUDE_DIR
        "${NVIDIA_USD_ROOT}/${NVIDIA_RELEASE_FOLDER_NAME}/include"
    DEBUG_INCLUDE_DIR
        "${NVIDIA_USD_ROOT}/${NVIDIA_DEBUG_FOLDER_NAME}/include"
    RELEASE_LIBRARY_DIR
        "${NVIDIA_USD_ROOT}/${NVIDIA_RELEASE_FOLDER_NAME}/lib"
    RELEASE_DLL_DIR
        "${NVIDIA_USD_ROOT}/${NVIDIA_RELEASE_FOLDER_NAME}/bin"
    DEBUG_LIBRARY_DIR
        "${NVIDIA_USD_ROOT}/${NVIDIA_DEBUG_FOLDER_NAME}/lib"
    DEBUG_DLL_DIR
        "${NVIDIA_USD_ROOT}/${NVIDIA_DEBUG_FOLDER_NAME}/bin"
    RELEASE_LIBRARIES
        ${NVIDIA_USD_RELEASE_LIBRARIES}
    DEBUG_LIBRARIES
        ${NVIDIA_USD_DEBUG_LIBRARIES}
    TARGET_NAMES
        ${NVIDIA_USD_TARGET_NAMES}
    TARGETS_IN_DLL_DIR
        tbb
)
# cmake-format: on

if(WIN32)
    # cmake-format: off
    add_prebuilt_project(
        RELEASE_INCLUDE_DIR
            "${PYTHON_ROOT}/include"
        DEBUG_INCLUDE_DIR
            "${PYTHON_ROOT}/include"
        RELEASE_LIBRARY_DIR
            "${PYTHON_ROOT}/libs"
        RELEASE_DLL_DIR
            "${PYTHON_ROOT}"
        DEBUG_LIBRARY_DIR
            "${PYTHON_ROOT}/libs"
        DEBUG_DLL_DIR
            "${PYTHON_ROOT}"
        RELEASE_LIBRARIES
            python312
        DEBUG_LIBRARIES
            python312
        TARGET_NAMES
            python312
    )
    # cmake-format: on
else()
    # cmake-format: off
    add_prebuilt_project(
        RELEASE_INCLUDE_DIR
            "${PYTHON_ROOT}/include/python3.12"
        DEBUG_INCLUDE_DIR
            "${PYTHON_ROOT}/include/python3.12"
        RELEASE_LIBRARY_DIR
            "${PYTHON_ROOT}/lib"
        DEBUG_LIBRARY_DIR
            "${PYTHON_ROOT}/lib"
        RELEASE_LIBRARIES
            python3.12
        DEBUG_LIBRARIES
            python3.12
        TARGET_NAMES
            python312
    )
    # cmake-format: on
endif()

# cmake-format: off
add_prebuilt_project_header_only(
    INCLUDE_DIR
        "${KIT_SDK_ROOT}/dev/fabric/include"
    TARGET_NAME
        fabric
)
# cmake-format: on

# cmake-format: off
add_prebuilt_project(
    RELEASE_INCLUDE_DIR
        "${CARB_ROOT}/include"
    DEBUG_INCLUDE_DIR
        "${CARB_ROOT}/include"
    RELEASE_LIBRARY_DIR
        "${CARB_ROOT}/_build/${CARB_PLATFORM_NAME}/${NVIDIA_RELEASE_FOLDER_NAME}"
    DEBUG_LIBRARY_DIR
        "${CARB_ROOT}/_build/${CARB_PLATFORM_NAME}/${NVIDIA_DEBUG_FOLDER_NAME}"
    RELEASE_LIBRARIES
        carb
    DEBUG_LIBRARIES
        carb
    TARGET_NAMES
        carb
)
# cmake-format: on

# cmake-format: off
add_prebuilt_project_header_only(
    INCLUDE_DIR
        "${KIT_SDK_ROOT}/dev/include"
    TARGET_NAME
        omni_kit
)
# cmake-format: on

# cmake-format: off
add_prebuilt_project_header_only(
    INCLUDE_DIR
        "${PYBIND11_ROOT}"
    TARGET_NAME
        pybind11
)
# cmake-format: on

# cmake-format: off
add_prebuilt_project_header_only(
    INCLUDE_DIR
        "${BOOST_PREPROCESSOR_ROOT}"
    TARGET_NAME
        boost_preprocessor
)
# cmake-format: on

# cmake-format: off
add_prebuilt_project_header_only(
    INCLUDE_DIR
        "${GSL_ROOT}"
    TARGET_NAME
        gsl
)
# cmake-format: on

# cmake-format: off
# omni.ui gives us access to DynamicTextureProvider.h
add_prebuilt_project(
    RELEASE_INCLUDE_DIR
        "${KIT_SDK_ROOT}/dev/include"
    DEBUG_INCLUDE_DIR
        "${KIT_SDK_ROOT}/dev/include"
    RELEASE_LIBRARY_DIR
        "${KIT_OMNI_UI_ROOT}/bin"
    DEBUG_LIBRARY_DIR
        "${KIT_OMNI_UI_ROOT}/bin"
    RELEASE_LIBRARIES
        omni.ui
    DEBUG_LIBRARIES
        omni.ui
    TARGET_NAMES
        omni_ui
)
# cmake-format: on

if(WIN32)
    set(NVIDIA_ADDITIONAL_LIBRARIES "${KIT_IMGUI_ROOT}/bin/imgui.dll")

    set_property(GLOBAL PROPERTY NVIDIA_ADDITIONAL_LIBRARIES_PROPERTY "${NVIDIA_ADDITIONAL_LIBRARIES}")
else()
    set(NVIDIA_ADDITIONAL_SEARCH_PATHS
        "$<TARGET_FILE_DIR:python312>"
        "$<TARGET_FILE_DIR:usd_usd>"
        "$<TARGET_FILE_DIR:carb>"
        # This is where freetype is located on Linux
        "${KIT_IMGUI_ROOT}/bin/deps")

    set(NVIDIA_ADDITIONAL_LINK_DIRECTORIES
        # This is where freetype is located on Linux. Needed by imgui which doesn't set its rpath properly
        "${KIT_IMGUI_ROOT}/bin/deps")

    set_property(GLOBAL PROPERTY NVIDIA_ADDITIONAL_SEARCH_PATHS_PROPERTY "${NVIDIA_ADDITIONAL_SEARCH_PATHS}")
    set_property(GLOBAL PROPERTY NVIDIA_ADDITIONAL_LINK_DIRECTORIES_PROPERTY "${NVIDIA_ADDITIONAL_LINK_DIRECTORIES}")
endif()
