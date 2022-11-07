include(Macros)

# cmake-format: off
add_external_project(
    PROJECT_NAME
        cesium-native
    LIBRARIES
        Cesium3DTilesSelection
        CesiumAsync
        CesiumGeometry
        CesiumGeospatial
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
        uriparser
        webpdecoder
    OPTIONS
        CESIUM_TESTS_ENABLED=OFF
        CESIUM_COVERAGE_ENABLED=OFF
    PROJECT_EXTERN_DIRECTORY
        "${PROJECT_SOURCE_DIR}/extern"
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

set(NVIDIA_BUILD_DIR "${PROJECT_SOURCE_DIR}/extern/nvidia/_build")
set(NVIDIA_USD_ROOT "${NVIDIA_BUILD_DIR}/target-deps/usd")
set(PYTHON_ROOT "${NVIDIA_BUILD_DIR}/target-deps/python")

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
    usdShade
    usdUtils
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

if(WIN32)
    # On Windows, Python is bundled with the USD package
    set(NVIDIA_USD_TARGET_NAMES ${NVIDIA_USD_TARGET_NAMES} python37)
    set(NVIDIA_USD_RELEASE_LIBRARIES ${NVIDIA_USD_RELEASE_LIBRARIES} python37)
    set(NVIDIA_USD_DEBUG_LIBRARIES ${NVIDIA_USD_DEBUG_LIBRARIES} python37)
endif()

# cmake-format: off
add_prebuilt_project(
    RELEASE_INCLUDE_DIR
        "${NVIDIA_USD_ROOT}/release/include"
    DEBUG_INCLUDE_DIR
        "${NVIDIA_USD_ROOT}/debug/include"
    RELEASE_LIBRARY_DIR
        "${NVIDIA_USD_ROOT}/release/lib"
    DEBUG_LIBRARY_DIR
        "${NVIDIA_USD_ROOT}/debug/lib"
    RELEASE_LIBRARIES
        ${NVIDIA_USD_RELEASE_LIBRARIES}
    DEBUG_LIBRARIES
        ${NVIDIA_USD_DEBUG_LIBRARIES}
    TARGET_NAMES
        ${NVIDIA_USD_TARGET_NAMES}
)
# cmake-format: on

# cmake-format: off
if (NOT WIN32)
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
endif()
# cmake-format: on
