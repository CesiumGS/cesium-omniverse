include(Macros)

glob_files(SOURCES "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")

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
        "${CMAKE_CURRENT_LIST_DIR}/include"
    LIBRARIES
        CesiumUsdSchemas
        Cesium3DTiles
        Cesium3DTilesSelection
        CesiumTilesetJsonReader
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
