include(Macros)

find_package(ovstage)

glob_files(SOURCES "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")

# cmake-format: off
setup_lib(
    TARGET_NAME
        CesiumOmniverseStage
    SOURCES
        ${SOURCES}
    INCLUDE_DIRS
        "${PROJECT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_LIST_DIR}/include"
    LIBRARIES
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
        ovstage::ovstage
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
