cmd /c cmake -B build-package -D CMAKE_BUILD_TYPE=Release CESIUM_OMNI_ENABLE_TESTS=OFF -D CESIUM_OMNI_ENABLE_DOCUMENTATION=OFF -D CESIUM_OMNI_ENABLE_SANITIZERS=OFF -D CESIUM_OMNI_ENABLE_LINTERS=OFF
cmd /c cmake --build build --config Release
cmd /c cmake --build build --target install --config Release
cmd /c cmake --build build --target package --config Release
