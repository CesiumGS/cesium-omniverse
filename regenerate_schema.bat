@echo off

set PROJECT_ROOT=%~dp0

set NVIDIA_USD_ROOT=%PROJECT_ROOT%\extern\nvidia\_build\target-deps\usd\release
set NVIDIA_USD_LIBS=%NVIDIA_USD_ROOT%\lib
set NVIDIA_USD_PYTHON_LIBS=%NVIDIA_USD_ROOT%\lib\python

set NVIDIA_PYTHON_ROOT=%PROJECT_ROOT%\extern\nvidia\_build\target-deps\python
set NVIDIA_PYTHON_EXECUTABLE=%NVIDIA_PYTHON_ROOT%\python.exe

set SCHEMA_INPUT_PATH=%PROJECT_ROOT%\exts\cesium.usd.plugins\schemas\cesium_schemas.usda
set SCHEMA_OUTPUT_PATH=%PROJECT_ROOT%\src\plugins\CesiumUsdSchemas\src\CesiumUsdSchemas

set PYTHONPATH=%NVIDIA_USD_PYTHON_LIBS%;%PYTHONPATH%
set PATH=%NVIDIA_USD_LIBS%;%PATH%

%NVIDIA_PYTHON_EXECUTABLE% -m pip install jinja2

:: Temporarily move module.cpp and moduleDeps.cpp out of the folder.
move %SCHEMA_OUTPUT_PATH%\module.cpp %PROJECT_ROOT%\module.cpp
move %SCHEMA_OUTPUT_PATH%\moduleDeps.cpp %PROJECT_ROOT%\moduleDeps.cpp

:: Clear out the old files.
del /s /q %SCHEMA_OUTPUT_PATH%\*

:: Generate the new files.
%NVIDIA_PYTHON_EXECUTABLE% %NVIDIA_USD_ROOT%\bin\usdGenSchema %SCHEMA_INPUT_PATH% %SCHEMA_OUTPUT_PATH%

:: Move the generatedSchema.usda and plugInfo.json files up.
move %SCHEMA_OUTPUT_PATH%\generatedSchema.usda %SCHEMA_OUTPUT_PATH%\..\..\generatedSchema.usda.in
move %SCHEMA_OUTPUT_PATH%\plugInfo.json %SCHEMA_OUTPUT_PATH%\..\..\plugInfo.json.in

:: Delete the Pixar junk from the first 23 lines of the generated code.
for %%f in (%SCHEMA_OUTPUT_PATH%\*.*) do (
    more +23 "%%f" > "%TEMP%\%%~nf%%~xf"
    move /y "%TEMP%\%%~nf%%~xf" "%%f" > nul
)

:: Move module.cpp and moduleDeps.cpp back.
move %PROJECT_ROOT%\module.cpp %SCHEMA_OUTPUT_PATH%\module.cpp
move %PROJECT_ROOT%\moduleDeps.cpp %SCHEMA_OUTPUT_PATH%\moduleDeps.cpp
