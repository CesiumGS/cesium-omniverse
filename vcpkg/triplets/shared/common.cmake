list(APPEND VCPKG_ENV_PASSTHROUGH "CESIUM_VCPKG_RELEASE_ONLY")

if(DEFINED ENV{CESIUM_VCPKG_RELEASE_ONLY} AND "$ENV{CESIUM_VCPKG_RELEASE_ONLY}")
  set(VCPKG_BUILD_TYPE "release")
endif()
