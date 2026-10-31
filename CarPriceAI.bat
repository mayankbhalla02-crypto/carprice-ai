@echo off
setlocal
cd /d "%~dp0"
title Car price AI
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 run.py
  goto done
)
where python >nul 2>nul
if %errorlevel%==0 (
  python run.py
  goto done
)
echo.
echo   Python 3 was not found. Opening the download page...
echo   Install it, tick "Add python.exe to PATH", then double-click this file again.
start "" https://www.python.org/downloads/
:done
echo.
pause
