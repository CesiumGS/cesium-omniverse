:: Initialize the MSVC C++ development environment before running the python script
:: This ensures that environment variables are populated and CMake can find cl.exe when using the Ninja generator
@echo off

:: Check if vswhere is in the path already (e.g. if installed with chocolatey)
where /q vswhere

if %ERRORLEVEL% equ 0 (
    :: Found vswhere, now get the full path
    for /f "delims=" %%i in ('where vswhere') do set vswhere_path="%%i"
) else (
    :: Otherwise look for vswhere in the Visual Studio Installer directory
    set vswhere_path="%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
)

if not exist %vswhere_path% (
    echo Could not find vswhere.exe
    exit /b 1
)

echo vswhere.exe path: %vswhere_path%

:: Find vsdevcmd.bat in the latest Visual Studio installation
:: See https://github.com/microsoft/vswhere/wiki/Find-VC
for /f "usebackq tokens=*" %%i in (`%vswhere_path% -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set vsdevcmd_path="%%i\Common7\Tools\vsdevcmd.bat"

if not exist %vsdevcmd_path% (
    echo "Could not find vsdevcmd.bat"
    exit /b 1
)

echo vsdevcmd.bat path: %vsdevcmd_path%

:: Call vsdevcmd.bat. This will start the developer command prompt and populate MSVC environment variables.
call %vsdevcmd_path% -arch=x64 -host_arch=x64

:: Now run our python script
python3 %~dp0\vscode_build.py %*
