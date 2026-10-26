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
        CesiumImage
        CesiumQuantizedMeshTerrain
        CesiumJsonReader
        CesiumJsonWriter
        CesiumGeospatial
        CesiumGeometry
        CesiumIonClient
        CesiumClientCommon
        CesiumCurl
        CesiumAsync
        CesiumUtility
        fabric
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
        usd_work
        tbb
        carb
        omni_kit
        omni_ui
        pybind11
        boost_preprocessor
        gsl
        python312
    ADDITIONAL_LIBRARIES
        "${ADDITIONAL_LIBRARIES}"
    ADDITIONAL_LINK_DIRECTORIES
        "${ADDITIONAL_LINK_DIRECTORIES}"
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
