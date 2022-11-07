@echo off

:Build
call "%~dp0tools\packman\python.bat" "%~dp0tools\repoman\build.py" --platform-target windows-x86_64 %*
if errorlevel 1 ( goto Error )

:Success
exit /b 0

:Error
exit /b %errorlevel%
