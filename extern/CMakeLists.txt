include(Macros)

# cmake-format: off
add_external_project(
    PROJECT_NAME
        cesium-native
    LIBRARIES
        CesiumIonClient
        Cesium3DTilesSelection
        CesiumAsync
        CesiumGeospatial
        CesiumGeometry
        CesiumGltf
        CesiumGltfReader
        CesiumJsonReader
        CesiumUtility
        async++
        draco
        ktx_read
        modp_b64
        s2geometry
        spdlog
        tinyxml2
        uriparser
        webpdecoder
    OPTIONS
        CESIUM_TESTS_ENABLED=OFF
        CESIUM_COVERAGE_ENABLED=OFF
    PROJECT_EXTERN_DIRECTORY
        "${PROJECT_SOURCE_DIR}/extern"
    EXPECTED_DEBUG_POSTFIX "d"
)
# cmake-format: on

if(UNIX)
    execute_process(COMMAND bash "${PROJECT_SOURCE_DIR}/extern/nvidia/prebuild.sh" RESULT_VARIABLE exit_code)
elseif(WIN32)
    execute_process(COMMAND cmd /C "${PROJECT_SOURCE_DIR}/extern/nvidia/prebuild.bat" RESULT_VARIABLE exit_code)
endif()

# cmake-format: off
if(exit_code AND NOT exit_code EQUAL 0)
    message(FATAL_ERROR "Gathering Nvidia libraries failed")
endif()
# cmake-format: on

if(UNIX)
    execute_process(COMMAND bash "${PROJECT_SOURCE_DIR}/extern/nvidia/link_app.sh")
elseif(WIN32)
    execute_process(COMMAND cmd /C "${PROJECT_SOURCE_DIR}/extern/nvidia/link_app.bat")
endif()

# Don't check for errors from the previous command because we can still proceed with the build
# without creating a symlink to Omniverse Code

if(WIN32)
    set(NVIDIA_PLATFORM_NAME "windows-x86_64")
elseif(UNIX AND NOT APPLE)
    set(NVIDIA_PLATFORM_NAME "linux-x86_64")
else()
    message(FATAL_ERROR "Only Windows and Linux are supported")
endif()

set(NVIDIA_RELEASE_FOLDER_NAME "release")

if(${USE_NVIDIA_RELEASE_LIBRARIES})
    set(NVIDIA_DEBUG_FOLDER_NAME "release")
else()
    set(NVIDIA_DEBUG_FOLDER_NAME "debug")
endif()

set(NVIDIA_BUILD_DIR "${PROJECT_SOURCE_DIR}/extern/nvidia/_build")
set(NVIDIA_USD_ROOT "${NVIDIA_BUILD_DIR}/target-deps/usd")
set(PYTHON_ROOT "${NVIDIA_BUILD_DIR}/target-deps/python")
set(USDRT_ROOT "${NVIDIA_BUILD_DIR}/target-deps/usdrt")
set(CARB_ROOT "${NVIDIA_BUILD_DIR}/target-deps/carb_sdk_plugins")
set(PYBIND11_ROOT "${NVIDIA_BUILD_DIR}/target-deps/pybind11")
set(CUDA_ROOT "${NVIDIA_BUILD_DIR}/target-deps/cuda")

set(NVIDIA_USD_LIBRARIES
    ar
    arch
    gf
    js
    kind
    ndr
    pcp
    plug
    sdf
    sdr
    tf
    trace
    usd
    usdGeom
    usdLux
    usdShade
    usdUI
    usdUtils
    usdVol
    vt
    work)

# Add base USD libraries
set(NVIDIA_USD_TARGET_NAMES ${NVIDIA_USD_LIBRARIES})
set(NVIDIA_USD_RELEASE_LIBRARIES ${NVIDIA_USD_LIBRARIES})
set(NVIDIA_USD_DEBUG_LIBRARIES ${NVIDIA_USD_LIBRARIES})

# Add TBB
set(NVIDIA_USD_TARGET_NAMES ${NVIDIA_USD_TARGET_NAMES} tbb)
set(NVIDIA_USD_RELEASE_LIBRARIES ${NVIDIA_USD_RELEASE_LIBRARIES} tbb)
set(NVIDIA_USD_DEBUG_LIBRARIES ${NVIDIA_USD_DEBUG_LIBRARIES} tbb_debug)

# Add boost python
set(NVIDIA_USD_TARGET_NAMES ${NVIDIA_USD_TARGET_NAMES} boost_python37)

if(WIN32)
    set(NVIDIA_USD_RELEASE_LIBRARIES ${NVIDIA_USD_RELEASE_LIBRARIES} boost_python37-vc141-mt-x64-1_68)
    set(NVIDIA_USD_DEBUG_LIBRARIES ${NVIDIA_USD_DEBUG_LIBRARIES} boost_python37-vc141-mt-gd-x64-1_68)
else()
    set(NVIDIA_USD_RELEASE_LIBRARIES ${NVIDIA_USD_RELEASE_LIBRARIES} boost_python37)
    set(NVIDIA_USD_DEBUG_LIBRARIES ${NVIDIA_USD_DEBUG_LIBRARIES} boost_python37)
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
    DEBUG_LIBRARY_DIR
        "${NVIDIA_USD_ROOT}/${NVIDIA_DEBUG_FOLDER_NAME}/lib"
    RELEASE_LIBRARIES
        ${NVIDIA_USD_RELEASE_LIBRARIES}
    DEBUG_LIBRARIES
        ${NVIDIA_USD_DEBUG_LIBRARIES}
    TARGET_NAMES
        ${NVIDIA_USD_TARGET_NAMES}
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
            python37
        DEBUG_LIBRARIES
            python37
        TARGET_NAMES
            python37
    )
    # cmake-format: on
else()
    # cmake-format: off
    add_prebuilt_project(
        RELEASE_INCLUDE_DIR
            "${PYTHON_ROOT}/include/python3.7m"
        DEBUG_INCLUDE_DIR
            "${PYTHON_ROOT}/include/python3.7m"
        RELEASE_LIBRARY_DIR
            "${PYTHON_ROOT}/lib"
        DEBUG_LIBRARY_DIR
            "${PYTHON_ROOT}/lib"
        RELEASE_LIBRARIES
            python3.7m
        DEBUG_LIBRARIES
            python3.7m
        TARGET_NAMES
            python37
    )
    # cmake-format: on
endif()

# cmake-format: off
add_prebuilt_project(
    RELEASE_INCLUDE_DIR
        "${USDRT_ROOT}/include"
    DEBUG_INCLUDE_DIR
        "${USDRT_ROOT}/include"
    RELEASE_LIBRARY_DIR
        "${USDRT_ROOT}/_build/${NVIDIA_PLATFORM_NAME}/${NVIDIA_RELEASE_FOLDER_NAME}"
    DEBUG_LIBRARY_DIR
        "${USDRT_ROOT}/_build/${NVIDIA_PLATFORM_NAME}/${NVIDIA_DEBUG_FOLDER_NAME}"
    RELEASE_LIBRARIES
        usdrt.scenegraph.plugin
    DEBUG_LIBRARIES
        usdrt.scenegraph.plugin
    TARGET_NAMES
        usdrt
)
# cmake-format: on

# cmake-format: off
add_prebuilt_project(
    RELEASE_INCLUDE_DIR
        "${CARB_ROOT}/include"
    DEBUG_INCLUDE_DIR
        "${CARB_ROOT}/include"
    RELEASE_LIBRARY_DIR
        "${CARB_ROOT}/_build/${NVIDIA_PLATFORM_NAME}/${NVIDIA_RELEASE_FOLDER_NAME}"
    DEBUG_LIBRARY_DIR
        "${CARB_ROOT}/_build/${NVIDIA_PLATFORM_NAME}/${NVIDIA_DEBUG_FOLDER_NAME}"
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
        "${PYBIND11_ROOT}"
    TARGET_NAME
        pybind11
)
# cmake-format: on

if(WIN32)
    # cmake-format: off
    add_prebuilt_project(
        RELEASE_INCLUDE_DIR
            "${CUDA_ROOT}"
        DEBUG_INCLUDE_DIR
            "${CUDA_ROOT}"
        RELEASE_LIBRARY_DIR
            "${CUDA_ROOT}/cuda/lib/x64"
        RELEASE_DLL_DIR
            "${CUDA_ROOT}/cuda/bin"
        DEBUG_LIBRARY_DIR
            "${CUDA_ROOT}/cuda/lib/x64"
        DEBUG_DLL_DIR
            "${CUDA_ROOT}/cuda/bin"
        RELEASE_LIBRARIES
            cudart
        DEBUG_LIBRARIES
            cudart
        RELEASE_DLL_LIBRARIES
            cudart64_110
        DEBUG_DLL_LIBRARIES
            cudart64_110
        TARGET_NAMES
            cudart
    )
    # cmake-format: on
else()
    # cmake-format: off
    add_prebuilt_project(
        RELEASE_INCLUDE_DIR
            "${CUDA_ROOT}"
        DEBUG_INCLUDE_DIR
            "${CUDA_ROOT}"
        RELEASE_LIBRARY_DIR
            "${CUDA_ROOT}/cuda/lib64"
        DEBUG_LIBRARY_DIR
            "${CUDA_ROOT}/cuda/lib64"
        RELEASE_LIBRARIES
            cudart
        DEBUG_LIBRARIES
            cudart
        TARGET_NAMES
            cudart
    )
    # cmake-format: on
endif()
