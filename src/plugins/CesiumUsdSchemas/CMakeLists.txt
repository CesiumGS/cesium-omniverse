include(Macros)

glob_files(SOURCES "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")

# cmake-format: off
setup_lib(
    TARGET_NAME
        CesiumUsdSchemas
    TYPE
        SHARED
    INCLUDE
        "${CMAKE_CURRENT_LIST_DIR}/include"
    SOURCES
        ${SOURCES}
    LIBRARIES
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
        work
        tbb
        boost_python37
        python37
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

add_custom_command(
    TARGET CesiumUsdSchemas
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_BINARY_DIR}/cesiumUsdSchemas.plugInfo.json.build"
            "$<TARGET_FILE_DIR:CesiumUsdSchemas>/plugins/CesiumUsdSchemas/resources/plugInfo.json"
    COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_BINARY_DIR}/cesiumUsdSchemas.generatedSchema.usda.build"
            "$<TARGET_FILE_DIR:CesiumUsdSchemas>/plugins/CesiumUsdSchemas/resources/generatedSchema.usda")
