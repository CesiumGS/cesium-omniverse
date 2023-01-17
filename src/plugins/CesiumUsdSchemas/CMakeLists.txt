include(Macros)

glob_files(SOURCES "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")

set(CESIUM_USD_SCHEMAS_CXX_FLAGS ${CESIUM_OMNI_CXX_FLAGS})

if(MSVC)
    # Disable whole program optimization and unreferenced function removal
    # Without these flags USD plugins won't work in Release mode
    # Related issue: https://github.com/PixarAnimationStudios/USD/issues/1095
    #
    # Also disable warning C4459:
    #  declaration of 'self' hides global declaration
    #  note: see declaration of 'boost::python::self_ns::self'
    set(CESIUM_USD_SCHEMAS_CXX_FLAGS ${CESIUM_USD_SCHEMAS_CXX_FLAGS} /GL- /Zc:inline- /wd4459)
endif()

set(CESIUM_USD_SCHEMAS_CXX_DEFINES ${CESIUM_OMNI_CXX_DEFINES} CESIUM_EXPORTS)

# cmake-format: off
setup_lib(
    TARGET_NAME
        CesiumUsdSchemas
    TYPE
        SHARED
    INCLUDE_DIRS
        "${CMAKE_CURRENT_LIST_DIR}/include"
    SOURCES
        ${SOURCES}
    LIBRARIES
        arch
        sdf
        tf
        usd
        tbb
        boost_python37
        python37
    CXX_FLAGS
        ${CESIUM_USD_SCHEMAS_CXX_FLAGS}
    CXX_FLAGS_DEBUG
        ${CESIUM_OMNI_CXX_FLAGS_DEBUG}
    CXX_DEFINES
        ${CESIUM_USD_SCHEMAS_CXX_DEFINES}
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
