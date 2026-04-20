@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "EASYICU_VERBOSE=0"
chcp 65001 >nul 2>nul

where py >nul 2>nul
if %ERRORLEVEL%==0 (
  py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul
  if %ERRORLEVEL%==0 (
    py -3 -X utf8 "%SCRIPT_DIR%scripts\\launch_easyicu.py" start %*
    set "EXIT_CODE=%ERRORLEVEL%"
    goto :done
  )
)

where python >nul 2>nul
if %ERRORLEVEL%==0 (
  python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul
  if %ERRORLEVEL%==0 (
    python -X utf8 "%SCRIPT_DIR%scripts\\launch_easyicu.py" start %*
    set "EXIT_CODE=%ERRORLEVEL%"
    goto :done
  )
)

echo Python 3.9+ was not found. Please install Python from https://www.python.org/downloads/
set "EXIT_CODE=1"

:done
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%
