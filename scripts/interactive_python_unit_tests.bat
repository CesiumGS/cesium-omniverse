@echo off
setlocal
set KIT_DIR=%~dp0\..\extern\nvidia\_build\target-deps\kit-sdk
call "%KIT_DIR%\kit.exe"  --enable omni.kit.test --dev --/app/enableStdoutOutput=0 --/exts/omni.kit.test/testExts/0='cesium.omniverse' --ext-folder "%KIT_DIR%/extscore" --ext-folder "%KIT_DIR%/exts" --ext-folder "%KIT_DIR%/apps" --ext-folder "%~dp0/../exts" --/exts/omni.kit.test/testExtApp="%~dp0/../apps/cesium.omniverse.dev.kit" --/exts/omni.kit.test/testExtOutputPath="%~dp0/../_testoutput" --portable-root "%KIT_DIR%/" --/telemetry/mode=test %*
