include(Macros)

glob_files(SOURCES "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")

set(IN_MEMORY_ASSET_RESOLVER_CXX_FLAGS ${CESIUM_OMNI_CXX_FLAGS})

if(MSVC)
    # Disable whole program optimization and unreferenced function removal
    # Without these flags USD plugins won't work in Release mode
    # Related issue: https://github.com/PixarAnimationStudios/USD/issues/1095
    set(IN_MEMORY_ASSET_RESOLVER_CXX_FLAGS ${IN_MEMORY_ASSET_RESOLVER_CXX_FLAGS} /GL- /Zc:inline-)
endif()

set(IN_MEMORY_ASSET_RESOLVER_CXX_DEFINES ${CESIUM_OMNI_CXX_DEFINES} AR_EXPORTS)

# cmake-format: off
setup_lib(
    TARGET_NAME
        InMemoryAssetResolver
    TYPE
        # InMemoryAssetResolver needs to be built as a SHARED library always so that it's picked up by the USD plugin system
        SHARED
    INCLUDE_DIRS
        "${CMAKE_CURRENT_LIST_DIR}/include"
    SOURCES
        ${SOURCES}
    LIBRARIES
        ar
        arch
        usd
        tbb
        tf
    CXX_FLAGS
        ${IN_MEMORY_ASSET_RESOLVER_CXX_FLAGS}
    CXX_FLAGS_DEBUG
        ${CESIUM_OMNI_CXX_FLAGS_DEBUG}
    CXX_DEFINES
        ${IN_MEMORY_ASSET_RESOLVER_CXX_DEFINES}
    CXX_DEFINES_DEBUG
        ${CESIUM_OMNI_CXX_DEFINES_DEBUG}
)
# cmake-format: on

# Copy plugInfo.json and mem.cesium into each build folder
add_custom_command(
    TARGET InMemoryAssetResolver
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_BINARY_DIR}/inMemoryAssetResolver.plugInfo.json.build"
            "$<TARGET_FILE_DIR:InMemoryAssetResolver>/plugins/InMemoryAssetResolver/resources/plugInfo.json"
    COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_BINARY_DIR}/mem.cesium"
            "$<TARGET_FILE_DIR:InMemoryAssetResolver>/mem.cesium")
