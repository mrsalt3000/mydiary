@echo off
rem ---------------------------------------------------------------------------
rem start-diary.bat - raise the local diary stack (Ollama + web :8000 + worker).
rem
rem This file may live anywhere: it uses absolute paths only and does not care
rem about the current directory, so a double click from a UNC path
rem (\\wsl.localhost\Ubuntu\home\mrsalt\projects\mydiary\scripts\windows\) works.
rem
rem The real work is done by scripts/service.sh inside WSL: it starts Ollama with
rem OLLAMA_KEEP_ALIVE=0, the web server without --reload, and the worker, all
rem through scripts/run-local.sh (which exports .env and the cuBLAS shim).
rem ---------------------------------------------------------------------------
chcp 65001 >nul
setlocal
set "DISTRO=Ubuntu"
set "REPO=/home/mrsalt/projects/mydiary"
set "URL=http://127.0.0.1:8000/"
rem Any argument suppresses the final pause - for callers that run this file
rem from a script instead of a double click.
set "PAUSE_AT_END=1"
if not "%~1"=="" set "PAUSE_AT_END="

echo === mydiary: starting the stack in WSL (%DISTRO%) ===
wsl.exe -d %DISTRO% -e bash -lc "cd %REPO% && scripts/service.sh start"
if errorlevel 1 goto failed

echo.
echo Ready: %URL%
start "" "%URL%"
echo.
if defined PAUSE_AT_END pause
exit /b 0

:failed
echo.
echo The stack did not start - see the output above.
echo Logs: \\wsl.localhost\%DISTRO%\home\mrsalt\.local\state\mydiary
echo.
if defined PAUSE_AT_END pause
exit /b 1
