# # Locate NVidia USD
# # lightweight find module

# # This module defines
# # NVidiaUSD_FOUND, if false, do not use
# # NVidiaUSD_INCLUDE_DIR, where to find the headers
# # NVidiaUSD_LIBRARY_DIR, where to find the library
# # NVidiaUSD_LIBRARIES
# #

# # the length of the Release and Debug lists need to match (same number of libraries)

# set(NVidiaUSD_LIB_NAMES_SHARED
#     ar
#     arch
#     gf
#     js
#     kind
#     pcp
#     plug
#     sdf
#     tf
#     trace
#     usd
#     usdGeom
#     vt
#     work
#     usdShade
#     usdLux
#     usdUtils
#     usdSkel
#     python37.lib)

# set(NVidiaUSD_LIB_NAMES_RELEASE ${NVidiaUSD_LIB_NAMES_SHARED} tbb boost_python37-vc141-mt-x64-1_68.lib)

# set(NVidiaUSD_LIB_NAMES_DEBUG ${NVidiaUSD_LIB_NAMES_SHARED} tbb_debug boost_python37-vc141-mt-gd-x64-1_68.lib)

# find_path(
#     NVidiaUSD_INCLUDE_DIR_RELEASE
#     NAMES "pxr/pxr.h"
#     PATHS ${NVidiaUSD_ROOT}/release/include
#           ${NVidiaUSD_ROOT}
#           /usr/local/include
#           /usr/include
#     NO_DEFAULT_PATH)

# find_path(
#     NVidiaUSD_INCLUDE_DIR_DEBUG
#     NAMES "pxr/pxr.h"
#     PATHS ${NVidiaUSD_ROOT}/debug/include
#           ${NVidiaUSD_ROOT}
#           /usr/local/include
#           /usr/include
#     NO_DEFAULT_PATH)

# # NVidiaUSD does not have a common include directory but separated by build config
# # as such, use a generator expression for the include dir variable

# set(NVidiaUSD_INCLUDE_DIR $<IF:$<CONFIG:Debug>,${NVidiaUSD_INCLUDE_DIR_DEBUG},${NVidiaUSD_INCLUDE_DIR_RELEASE}>)

# mark_as_advanced(NVidiaUSD_INCLUDE_DIR_RELEASE NVidiaUSD_INCLUDE_DIR_DEBUG)

# # pull the USD version

# if(EXISTS ${NVidiaUSD_INCLUDE_DIR_RELEASE})
#     foreach(_usd_comp MAJOR MINOR PATCH)
#         file(STRINGS "${NVidiaUSD_INCLUDE_DIR_RELEASE}/pxr/pxr.h" _usd_tmp REGEX "#define PXR_${_usd_comp}_VERSION .*$")
#         string(
#             REGEX MATCHALL
#                   "[0-9]+"
#                   USD_${_usd_comp}_VERSION
#                   ${_usd_tmp})
#     endforeach()
#     set(USD_VERSION ${USD_MAJOR_VERSION}.${USD_MINOR_VERSION}.${USD_PATCH_VERSION})
#     math(EXPR PXR_VERSION "${USD_MAJOR_VERSION} * 10000 + ${USD_MINOR_VERSION} * 100 + ${USD_PATCH_VERSION}")
# endif()

# find_path(
#     NVidiaUSD_LIBRARY_DIR_RELEASE
#     NAMES "usd.lib" "libusd.so"
#     PATHS ${NVidiaUSD_ROOT}/release/lib
#           ${NVidiaUSD_ROOT}
#           /usr/local/include
#           /usr/include
#     NO_DEFAULT_PATH)

# find_path(
#     NVidiaUSD_LIBRARY_DIR_DEBUG
#     NAMES "usd.lib" "libusd.so"
#     PATHS ${NVidiaUSD_ROOT}/debug/lib
#           ${NVidiaUSD_ROOT}
#           /usr/local/include
#           /usr/include
#     NO_DEFAULT_PATH)

# # set as an generator expression so it can have different uses

# set(NVidiaUSD_LIBRARY_DIR $<IF:$<CONFIG:Debug>,${NVidiaUSD_LIBRARY_DIR_DEBUG},${NVidiaUSD_LIBRARY_DIR_RELEASE}>)

# mark_as_advanced(NVidiaUSD_LIBRARY_DIR_RELEASE NVidiaUSD_LIBRARY_DIR_DEBUG)

# foreach(
#     dl
#     rl
#     IN
#     ZIP_LISTS
#     NVidiaUSD_LIB_NAMES_DEBUG
#     NVidiaUSD_LIB_NAMES_RELEASE)
#     unset(_nvusd_dl_name CACHE)
#     unset(_nvusd_rl_name CACHE)

#     find_library(
#         _nvusd_dl_name
#         NAMES ${dl}
#         PATHS ${NVidiaUSD_LIBRARY_DIR_DEBUG})

#     find_library(
#         _nvusd_rl_name
#         NAMES ${rl}
#         PATHS ${NVidiaUSD_LIBRARY_DIR_RELEASE})

#     list(
#         APPEND
#         NVidiaUSD_LIBRARIES
#         debug
#         ${_nvusd_dl_name}
#         optimized
#         ${_nvusd_rl_name})

#     unset(_nvusd_dl_name CACHE)
#     unset(_nvusd_rl_name CACHE)
# endforeach()

# unset(NVidiaUSD_LIB_NAMES_SHARED CACHE)
# unset(NVidiaUSD_LIB_NAMES_RELEASE CACHE)
# unset(NVidiaUSD_LIB_NAMES_DEBUG CACHE)

# include(FindPackageHandleStandardArgs)

# find_package_handle_standard_args(
#     NVidiaUSD
#     REQUIRED_VARS
#         NVidiaUSD_INCLUDE_DIR
#         NVidiaUSD_LIBRARY_DIR
#         NVidiaUSD_LIBRARIES
#         USD_VERSION
#         PXR_VERSION
#     VERSION_VAR USD_VERSION)
