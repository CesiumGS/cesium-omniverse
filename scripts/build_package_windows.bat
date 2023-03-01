@echo off

rmdir /s /q build-package

if %errorlevel% neq 0 exit /b %errorlevel%

cmd /c cmake -B build-package -D CMAKE_CONFIGURATION_TYPES=Release -D CESIUM_OMNI_ENABLE_TESTS=OFF -D CESIUM_OMNI_ENABLE_DOCUMENTATION=OFF -D CESIUM_OMNI_ENABLE_SANITIZERS=OFF -D CESIUM_OMNI_ENABLE_LINTERS=OFF

if %errorlevel% neq 0 exit /b %errorlevel%

cmd /c cmake --build build-package --config Release

if %errorlevel% neq 0 exit /b %errorlevel%

cmd /c cmake --build build-package --target install --config Release

if %errorlevel% neq 0 exit /b %errorlevel%

cmd /c cmake --build build-package --target package --config Release

if %errorlevel% neq 0 exit /b %errorlevel%
