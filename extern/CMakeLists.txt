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
        csprng
        draco
        ktx_read
        modp_b64
        s2geometry
        spdlog
        tinyxml2
        uriparser
        webpdecoder
        turbojpeg
        meshoptimizer
    OPTIONS
        CESIUM_TESTS_ENABLED=OFF
        CESIUM_COVERAGE_ENABLED=OFF
        CESIUM_TRACING_ENABLED=${CESIUM_OMNI_ENABLE_TRACING}
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

# Add a symlink to USD Composer (create) so that we can use its extensions (e.g. omni.kit.window.material_graph) in our internal applications
# Don't check for errors because we can still proceed with the build
if(UNIX)
    execute_process(COMMAND bash "${PROJECT_SOURCE_DIR}/extern/nvidia/link_app.sh --app create")
elseif(WIN32)
    execute_process(COMMAND cmd /C "${PROJECT_SOURCE_DIR}/extern/nvidia/link_app.bat --app create")
endif()

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
set(KIT_SDK_ROOT "${NVIDIA_BUILD_DIR}/target-deps/kit-sdk")
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
set(NVIDIA_USD_TARGET_NAMES ${NVIDIA_USD_TARGET_NAMES} boost_python310)

if(WIN32)
    set(NVIDIA_USD_RELEASE_LIBRARIES ${NVIDIA_USD_RELEASE_LIBRARIES} boost_python310-vc142-mt-x64-1_76)
    set(NVIDIA_USD_DEBUG_LIBRARIES ${NVIDIA_USD_DEBUG_LIBRARIES} boost_python310-vc142-mt-gd-x64-1_76)
else()
    set(NVIDIA_USD_RELEASE_LIBRARIES ${NVIDIA_USD_RELEASE_LIBRARIES} boost_python310)
    set(NVIDIA_USD_DEBUG_LIBRARIES ${NVIDIA_USD_DEBUG_LIBRARIES} boost_python310)
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
            python310
        DEBUG_LIBRARIES
            python310
        TARGET_NAMES
            python310
    )
    # cmake-format: on
else()
    # cmake-format: off
    add_prebuilt_project(
        RELEASE_INCLUDE_DIR
            "${PYTHON_ROOT}/include/python3.10"
        DEBUG_INCLUDE_DIR
            "${PYTHON_ROOT}/include/python3.10"
        RELEASE_LIBRARY_DIR
            "${PYTHON_ROOT}/lib"
        DEBUG_LIBRARY_DIR
            "${PYTHON_ROOT}/lib"
        RELEASE_LIBRARIES
            python3.10
        DEBUG_LIBRARIES
            python3.10
        TARGET_NAMES
            python310
    )
    # cmake-format: on
endif()

# cmake-format: off
add_prebuilt_project(
    RELEASE_INCLUDE_DIR
        "${USDRT_ROOT}/include/usdrt_only"
    DEBUG_INCLUDE_DIR
        "${USDRT_ROOT}/include/usdrt_only"
    RELEASE_LIBRARY_DIR
        "${USDRT_ROOT}/_build/${NVIDIA_PLATFORM_NAME}/${NVIDIA_RELEASE_FOLDER_NAME}/usdrt_only"
    DEBUG_LIBRARY_DIR
        "${USDRT_ROOT}/_build/${NVIDIA_PLATFORM_NAME}/${NVIDIA_DEBUG_FOLDER_NAME}/usdrt_only"
    RELEASE_LIBRARIES
        omni.fabric.plugin
    DEBUG_LIBRARIES
        omni.fabric.plugin
    TARGET_NAMES
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

# cmake-format: off
# omni.ui gives us access to DynamicTextureProvider.h
add_prebuilt_project(
    RELEASE_INCLUDE_DIR
        "${KIT_SDK_ROOT}/dev/include"
    DEBUG_INCLUDE_DIR
        "${KIT_SDK_ROOT}/dev/include"
    RELEASE_LIBRARY_DIR
        "${KIT_SDK_ROOT}/exts/omni.ui/bin"
    DEBUG_LIBRARY_DIR
        "${KIT_SDK_ROOT}/exts/omni.ui/bin"
    RELEASE_LIBRARIES
        omni.ui
    DEBUG_LIBRARIES
        omni.ui
    TARGET_NAMES
        omni_ui
)
# cmake-format: on

if(WIN32)
    set(NVIDIA_ADDITIONAL_LIBRARIES
        "${PROJECT_SOURCE_DIR}/extern/nvidia/_build/target-deps/kit-sdk/exts/omni.kit.renderer.imgui/bin/imgui.dll"
        "${PROJECT_SOURCE_DIR}/extern/nvidia/_build/target-deps/kit-sdk/exts/omni.kit.renderer.imgui/bin/deps/freetype.dll"
    )

    set_property(GLOBAL PROPERTY NVIDIA_ADDITIONAL_LIBRARIES_PROPERTY "${NVIDIA_ADDITIONAL_LIBRARIES}")
else()
    set(NVIDIA_ADDITIONAL_SEARCH_PATHS
        "$<TARGET_FILE_DIR:python310>"
        "$<TARGET_FILE_DIR:usd>"
        "$<TARGET_FILE_DIR:carb>"
        # This is where freetype is located on Linux
        "${PROJECT_SOURCE_DIR}/extern/nvidia/_build/target-deps/kit-sdk/exts/omni.kit.renderer.imgui/bin/deps")

    set(NVIDIA_ADDITIONAL_LINK_DIRECTORIES
        # This is where freetype is located on Linux. Needed by imgui which doesn't set its rpath properly
        "${PROJECT_SOURCE_DIR}/extern/nvidia/_build/target-deps/kit-sdk/exts/omni.kit.renderer.imgui/bin/deps")

    set_property(GLOBAL PROPERTY NVIDIA_ADDITIONAL_SEARCH_PATHS_PROPERTY "${NVIDIA_ADDITIONAL_SEARCH_PATHS}")
    set_property(GLOBAL PROPERTY NVIDIA_ADDITIONAL_LINK_DIRECTORIES_PROPERTY "${NVIDIA_ADDITIONAL_LINK_DIRECTORIES}")
endif()
