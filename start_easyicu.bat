@echo off
setlocal
set "SCRIPT_DIR=%~dp0"

where py >nul 2>nul
if %ERRORLEVEL%==0 (
  py -3 "%SCRIPT_DIR%launch_easyicu.py" start %*
  set "EXIT_CODE=%ERRORLEVEL%"
  goto :done
)

where python >nul 2>nul
if %ERRORLEVEL%==0 (
  python "%SCRIPT_DIR%launch_easyicu.py" start %*
  set "EXIT_CODE=%ERRORLEVEL%"
  goto :done
)

echo Python 3.9+ was not found. Please install Python from https://www.python.org/downloads/
set "EXIT_CODE=1"

:done
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%
