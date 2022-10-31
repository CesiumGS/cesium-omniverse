# Locate Omniverse Connect Sample Directory

# Users may adjust the behaviors of this module by modifying these variables.
#     OmniverseConnectSample_ROOT - install location
# This module defines
#     OmniverseConnectSample_FOUND - if false, do not use
#     OmniverseConnectSample_DIR - where to find the headers
#

set(OmniverseConnectSample_VERSION "200.0.0")

find_path(
    OmniverseConnectSample_DIR
    NAMES "run_omniSimpleSensor.bat" "run_omniSimpleSensor.sh"
    PATHS ${OmniverseConnectSample_ROOT} $ENV{LOCALAPPDATA}/ov/pkg $ENV{HOME}/.local/share/ov/pkg
    PATH_SUFFIXES connectsample-${OmniverseConnectSample_VERSION}
    NO_DEFAULT_PATH)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(
    OmniverseConnectSample
    REQUIRED_VARS OmniverseConnectSample_DIR
    VERSION_VAR OmniverseConnectSample_VERSION)
