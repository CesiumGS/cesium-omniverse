include(Macros)

if(MSVC)
    set(SSL_LIBRARY_NAME "libssl")
    set(CRYPTO_LIBRARY_NAME "libcrypto")
    set(WEBP_LIBRARY_NAME "libwebp")
    set(WEBPDECODER_LIBRARY_NAME "libwebpdecoder")
    set(WEBPDEMUX_LIBRARY_NAME "libwebpdemux")
    set(WEBPMUX_LIBRARY_NAME "libwebpmux")
    set(SHARPYUV_LIBRARY_NAME "libsharpyuv")
    set(ZNG_LIBRARY_NAME "zlibstatic-ng")
else()
    set(SSL_LIBRARY_NAME "ssl")
    set(CRYPTO_LIBRARY_NAME "crypto")
    set(WEBP_LIBRARY_NAME "webp")
    set(WEBPDECODER_LIBRARY_NAME "webpdecoder")
    set(WEBPDEMUX_LIBRARY_NAME "webpdemux")
    set(WEBPMUX_LIBRARY_NAME "webpmux")
    set(SHARPYUV_LIBRARY_NAME "sharpyuv")
    set(ZNG_LIBRARY_NAME "z-ng")
endif()

# cmake-format: off
add_external_project(
    PROJECT_NAME
        cesium-native
    LIBRARIES
        Cesium3DTilesSelection
        Cesium3DTilesReader
        Cesium3DTilesContent
        CesiumRasterOverlays
        CesiumGltfReader
        CesiumGltfContent
        CesiumGltf
        CesiumQuantizedMeshTerrain
        CesiumJsonReader
        CesiumGeospatial
        CesiumGeometry
        CesiumIonClient
        CesiumAsync
        CesiumUtility
        async++
        draco
        glm
        jpeg
        ktx
        libmodpbase64
        meshoptimizer
        s2
        absl_cordz_sample_token
        absl_die_if_null
        absl_failure_signal_handler
        absl_flags_parse
        absl_log_flags
        absl_log_initialize
        absl_periodic_sampler
        absl_random_internal_distribution_test_util
        absl_scoped_set_env
        absl_statusor
        absl_bad_any_cast_impl
        absl_log_internal_check_op
        absl_leak_check
        absl_flags_usage
        absl_flags_usage_internal
        absl_flags_internal
        absl_flags_reflection
        absl_flags_config
        absl_flags_program_name
        absl_flags_private_handle_accessor
        absl_flags_commandlineflag
        absl_flags_commandlineflag_internal
        absl_flags_marshalling
        absl_log_internal_conditions
        absl_log_internal_message
        absl_examine_stack
        absl_log_internal_nullguard
        absl_log_internal_format
        absl_log_internal_proto
        absl_log_internal_log_sink_set
        absl_log_globals
        absl_vlog_config_internal
        absl_log_internal_fnmatch
        absl_log_internal_globals
        absl_log_sink
        absl_log_entry
        absl_raw_hash_set
        absl_hash
        absl_city
        absl_low_level_hash
        absl_hashtablez_sampler
        absl_random_distributions
        absl_random_seed_sequences
        absl_random_internal_pool_urbg
        absl_random_internal_randen
        absl_random_internal_randen_hwaes
        absl_random_internal_randen_hwaes_impl
        absl_random_internal_randen_slow
        absl_random_internal_platform
        absl_random_internal_seed_material
        absl_random_seed_gen_exception
        absl_status
        absl_cord
        absl_cordz_info
        absl_cordz_handle
        absl_cordz_functions
        absl_exponential_biased
        absl_synchronization
        absl_graphcycles_internal
        absl_kernel_timeout_internal
        absl_time
        absl_civil_time
        absl_time_zone
        absl_cord_internal
        absl_crc_cord_state
        absl_crc32c
        absl_crc_internal
        absl_crc_cpu_detect
        absl_stacktrace
        absl_bad_optional_access
        absl_str_format_internal
        absl_strerror
        absl_symbolize
        absl_strings
        absl_strings_internal
        absl_int128
        absl_string_view
        absl_throw_delegate
        absl_debugging_internal
        absl_malloc_internal
        absl_demangle_internal
        absl_base
        absl_spinlock_wait
        absl_bad_variant_access
        absl_raw_logging_internal
        absl_log_severity
        ${SSL_LIBRARY_NAME}
        ${CRYPTO_LIBRARY_NAME}
        spdlog
        fmt
        sqlite3
        tinyxml2
        turbojpeg
        uriparser
        ${WEBP_LIBRARY_NAME}
        ${WEBPDECODER_LIBRARY_NAME}
        ${WEBPDEMUX_LIBRARY_NAME}
        ${WEBPMUX_LIBRARY_NAME}
        astcenc-avx2-static
        ${SHARPYUV_LIBRARY_NAME}
        ${ZNG_LIBRARY_NAME}
        zstd
    OPTIONS
        CESIUM_TESTS_ENABLED=OFF
        CESIUM_COVERAGE_ENABLED=OFF
        CESIUM_TRACING_ENABLED=${CESIUM_OMNI_ENABLE_TRACING}
        VCPKG_OVERLAY_PORTS=${VCPKG_OVERLAY_PORTS}
        VCPKG_OVERLAY_TRIPLETS=${VCPKG_OVERLAY_TRIPLETS}
        VCPKG_TRIPLET=${VCPKG_TRIPLET}
        EZVCPKG_BASEDIR=${EZVCPKG_BASEDIR}
    PROJECT_EXTERN_DIRECTORY
        "${PROJECT_SOURCE_DIR}/extern"
)
# cmake-format: on

if(NOT ${USE_NVIDIA_RELEASE_LIBRARIES})
    execute_process(COMMAND "${Python3_EXECUTABLE}" "${SCRIPTS_DIRECTORY}/copy_from_dir.py" "*.user.xml"
                            "${PROJECT_SOURCE_DIR}/extern/nvidia/debug-deps" "${PROJECT_SOURCE_DIR}/extern/nvidia/deps")
endif()

if(WIN32)
    set(NVIDIA_PLATFORM_NAME "windows-x86_64")
elseif(UNIX AND NOT APPLE)
    set(NVIDIA_PLATFORM_NAME "linux-x86_64")
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

if(WIN32)
    set(NVIDIA_USD_DEBUG_LIBRARIES ${NVIDIA_USD_DEBUG_LIBRARIES} tbb_debug)
else()
    set(NVIDIA_USD_DEBUG_LIBRARIES ${NVIDIA_USD_DEBUG_LIBRARIES} tbb)
endif()

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
    set(NVIDIA_ADDITIONAL_LIBRARIES
        "${KIT_IMGUI_ROOT}/bin/imgui.dll"
        "${KIT_IMGUI_ROOT}/bin/deps/freetype.dll"
    )

    set_property(GLOBAL PROPERTY NVIDIA_ADDITIONAL_LIBRARIES_PROPERTY "${NVIDIA_ADDITIONAL_LIBRARIES}")
else()
    set(NVIDIA_ADDITIONAL_SEARCH_PATHS
        "$<TARGET_FILE_DIR:python310>"
        "$<TARGET_FILE_DIR:usd>"
        "$<TARGET_FILE_DIR:carb>"
        # This is where freetype is located on Linux
        "${KIT_IMGUI_ROOT}/bin/deps")

    set(NVIDIA_ADDITIONAL_LINK_DIRECTORIES
        # This is where freetype is located on Linux. Needed by imgui which doesn't set its rpath properly
        "${KIT_IMGUI_ROOT}/bin/deps")

    set_property(GLOBAL PROPERTY NVIDIA_ADDITIONAL_SEARCH_PATHS_PROPERTY "${NVIDIA_ADDITIONAL_SEARCH_PATHS}")
    set_property(GLOBAL PROPERTY NVIDIA_ADDITIONAL_LINK_DIRECTORIES_PROPERTY "${NVIDIA_ADDITIONAL_LINK_DIRECTORIES}")
endif()
