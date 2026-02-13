# On Linux you may see a warning like:
#
# Warning: Unused direct dependencies:
#         cesium-omniverse/extern/nvidia/_build/target-deps/usd/release/lib/libarch.so
#         cesium-omniverse/extern/nvidia/_build/target-deps/usd/release/lib/libjs.so
#         cesium-omniverse/extern/nvidia/_build/target-deps/usd/release/lib/libkind.so
#         cesium-omniverse/extern/nvidia/_build/target-deps/usd/release/lib/libndr.so
#         cesium-omniverse/extern/nvidia/_build/target-deps/usd/release/lib/libpcp.so
#         cesium-omniverse/extern/nvidia/_build/target-deps/usd/release/lib/libplug.so
#         cesium-omniverse/extern/nvidia/_build/target-deps/usd/release/lib/libsdr.so
#         cesium-omniverse/extern/nvidia/_build/target-deps/usd/release/lib/libtrace.so
#
# This is because some USD libraries are referenced by other USD libraries but are not direct depenencies of
# Cesium Omniverse. The reason we include them in the list below is because on Windows this allows them to be
# copied to the bin folder when TARGET_RUNTIME_DLLS is used.

include(Macros)

glob_files(SOURCES "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")

get_property(ADDITIONAL_LIBRARIES GLOBAL PROPERTY NVIDIA_ADDITIONAL_LIBRARIES_PROPERTY)
get_property(ADDITIONAL_LINK_DIRECTORIES GLOBAL PROPERTY NVIDIA_ADDITIONAL_LINK_DIRECTORIES_PROPERTY)

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
setup_lib(
    TARGET_NAME
        CesiumOmniverseCore
    TYPE
        # CesiumOmniverseCore needs to be built as a STATIC library always so that it can be tested.
        # Otherwise if it was built as a shared library we would have to add dllexport attributes to almost everything.
        STATIC
    SOURCES
        ${SOURCES}
    INCLUDE_DIRS
        # Stub headers for rtx_plugins (not publicly available)
        "${CMAKE_CURRENT_LIST_DIR}/include/stubs"
        "${CMAKE_CURRENT_LIST_DIR}/include"
    LIBRARIES
        CesiumUsdSchemas
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
        # Already linking OpenSSL from CURL (from Conan)
        # ${SSL_LIBRARY_NAME}
        # ${CRYPTO_LIBRARY_NAME}
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
        CURL::libcurl
        stb::stb
        ZLIB::ZLIB
        fabric
        usd_ar
        usd_arch
        usd_gf
        usd_js
        usd_kind
        usd_ndr
        usd_pcp
        usd_plug
        usd_sdf
        usd_sdr
        usd_tf
        usd_trace
        usd_usd
        usd_usdGeom
        usd_usdLux
        usd_usdShade
        usd_usdUI
        usd_usdUtils
        usd_usdVol
        usd_vt
        usd_work
        usd_boost
        tbb
        carb
        omni_kit
        omni_ui
        pybind11
        python312
    ADDITIONAL_LIBRARIES
        "${ADDITIONAL_LIBRARIES}"
    ADDITIONAL_LINK_DIRECTORIES
        "${ADDITIONAL_LINK_DIRECTORIES}"
    DEPENDENCIES
        cesium-native-external
    CXX_FLAGS
        ${CESIUM_OMNI_CXX_FLAGS}
    CXX_FLAGS_DEBUG
        ${CESIUM_OMNI_CXX_FLAGS_DEBUG}
    CXX_DEFINES
        ${CESIUM_OMNI_CXX_DEFINES}
    CXX_DEFINES_DEBUG
        ${CESIUM_OMNI_CXX_DEFINES_DEBUG}
)
# cmake-format: on
