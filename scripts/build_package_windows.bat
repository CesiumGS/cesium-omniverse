@echo off

rmdir /s /q build-package
cmd /c cmake -B build-package -D CMAKE_CONFIGURATION_TYPES=Release -D CESIUM_OMNI_ENABLE_TESTS=OFF -D CESIUM_OMNI_ENABLE_DOCUMENTATION=OFF -D CESIUM_OMNI_ENABLE_SANITIZERS=OFF -D CESIUM_OMNI_ENABLE_LINTERS=OFF
cmd /c cmake --build build-package --config Release
cmd /c cmake --build build-package --target install --config Release
cmd /c cmake --build build-package --target package --config Release
