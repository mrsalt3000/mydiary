@echo off
rem ---------------------------------------------------------------------------
rem stop-diary.bat - stop the local diary stack (worker, web, Ollama).
rem
rem Counterpart of start-diary.bat; the work is done by scripts/service.sh stop
rem inside WSL. It also stops an Ollama server that was raised by hand (the
rem usual case documented in docs/session-handoff.md section 2), for which no
rem pid file exists.
rem
rem This file may live anywhere - it uses absolute paths only.
rem ---------------------------------------------------------------------------
chcp 65001 >nul
setlocal
set "DISTRO=Ubuntu"
set "REPO=/home/mrsalt/projects/mydiary"
rem Any argument suppresses the final pause - for callers that run this file
rem from a script instead of a double click.
set "PAUSE_AT_END=1"
if not "%~1"=="" set "PAUSE_AT_END="

echo === mydiary: stopping the stack in WSL (%DISTRO%) ===
wsl.exe -d %DISTRO% -e bash -lc "cd %REPO% && scripts/service.sh stop"
if errorlevel 1 goto failed

echo.
echo Stopped.
echo.
if defined PAUSE_AT_END pause
exit /b 0

:failed
echo.
echo Stop reported an error - see the output above.
echo Logs: \\wsl.localhost\%DISTRO%\home\mrsalt\.local\state\mydiary
echo.
if defined PAUSE_AT_END pause
exit /b 1
