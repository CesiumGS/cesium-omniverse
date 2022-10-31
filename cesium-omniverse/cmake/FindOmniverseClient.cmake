# Locate Omniverse Client
# lightweight find module

# This module defines
# OmniverseClient_FOUND, if false, do not use
# OmniverseClient_INCLUDE_DIR, where to find the headers
# OmniverseClient_LIBRARY_DIR
# OmniverseClient_LIBRARIES
#

# include directory is shared for both debug and release

find_path(
    OmniverseClient_INCLUDE_DIR
    NAMES "OmniClient.h"
    PATHS ${OmniverseClient_ROOT}/include
    NO_DEFAULT_PATH)

find_path(
    OmniverseClient_LIBRARY_DIR_RELEASE
    NAMES "omniclient.lib" "libomniclient.so"
    PATHS ${OmniverseClient_ROOT}/release
    NO_DEFAULT_PATH)

find_path(
    OmniverseClient_LIBRARY_DIR_DEBUG
    NAMES "omniclient.lib" "libomniclient.so"
    PATHS ${OmniverseClient_ROOT}/debug
    NO_DEFAULT_PATH)

# Created as a generator expressed so it can more useful
set(OmniverseClient_LIBRARY_DIR
    $<IF:$<CONFIG:Debug>,${OmniverseClient_LIBRARY_DIR_DEBUG},${OmniverseClient_LIBRARY_DIR_RELEASE}>)

mark_as_advanced(OmniverseClient_LIBRARY_DIR_RELEASE OmniverseClient_LIBRARY_DIR_DEBUG)

find_library(
    OmniverseClient_LIBRARIES_RELEASE
    NAMES "omniclient.lib" "libomniclient.so"
    PATHS ${OmniverseClient_LIBRARY_DIR_RELEASE})

find_library(
    OmniverseClient_LIBRARIES_DEBUG
    NAMES "omniclient.lib" "libomniclient.so"
    PATHS ${OmniverseClient_LIBRARY_DIR_DEBUG})

set(OmniverseClient_LIBRARIES
    debug
    "${OmniverseClient_LIBRARIES_DEBUG}"
    optimized
    "${OmniverseClient_LIBRARIES_RELEASE}")

mark_as_advanced(OmniverseClient_LIBRARIES_RELEASE OmniverseClient_LIBRARIES_DEBUG)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OmniverseClient REQUIRED_VARS OmniverseClient_INCLUDE_DIR OmniverseClient_LIBRARY_DIR
                                                                OmniverseClient_LIBRARIES)
