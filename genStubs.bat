@echo off

:: ********************************************************************
:: Note: This only works in kit >=105 due to something with python versioning
:: ********************************************************************

:: Get the nvidia python executable path, it has a few modules needed by the source
:: code we're stubbing, so those imports need to be valid

set PROJECT_ROOT=%~dp0

set NVIDIA_USD_ROOT=%PROJECT_ROOT%\extern\nvidia\_build\target-deps\usd\release
set NVIDIA_USD_PYTHON_LIBS=%NVIDIA_USD_ROOT%\lib\python

set NVIDIA_PYTHON_ROOT=%PROJECT_ROOT%\extern\nvidia\_build\target-deps\python
set NVIDIA_PYTHON_EXECUTABLE=%NVIDIA_PYTHON_ROOT%\python.exe

set FLAT_LIBRARIES_DIR=%TEMP%\CesiumOmniverseFlatLibs
set CESIUM_OMNI_STUB_PATH=%PROJECT_ROOT%\exts\cesium.omniverse\cesium\omniverse\bindings\CesiumOmniversePythonBindings.pyi
set CESIUM_USD_STUB_PATH=%PROJECT_ROOT%\exts\cesium.usd.plugins\cesium\usd\plugins\CesiumUsdSchemas\__init__.pyi

set PYTHONPATH=%NVIDIA_USD_PYTHON_LIBS%;%PYTHONPATH%

echo "Ensuring mypy is installed"
%NVIDIA_PYTHON_EXECUTABLE% -m pip install mypy

echo "Building lib files flat in temp dir"
cmake -B build
cmake --build build --config Release --parallel 8
cmake --install build --config Release --component library --prefix %FLAT_LIBRARIES_DIR%

:: To find the imports mypy has to be run from the same dir as the object files
cd %FLAT_LIBRARIES_DIR%\lib

:: We call mypy in this strange way to ensure the correct nvidia python executable is used
echo "Generating stubs"
%NVIDIA_PYTHON_EXECUTABLE% -c "from mypy import stubgen; stubgen.main()" -m CesiumOmniversePythonBindings -v
%NVIDIA_PYTHON_EXECUTABLE% -c "from mypy import stubgen; stubgen.main()" -m _CesiumUsdSchemas -v

echo "Copying stubs"
copy out\CesiumOmniversePythonBindings.pyi %CESIUM_OMNI_STUB_PATH%
copy out\_CesiumUsdSchemas.pyi %CESIUM_USD_STUB_PATH%

echo "Formatting stubs"
black %CESIUM_OMNI_STUB_PATH%
black %CESIUM_USD_STUB_PATH%

echo "Cleaning up temp dir"
cd %PROJECT_ROOT%
rmdir /s /q %FLAT_LIBRARIES_DIR%
