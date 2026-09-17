@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
set LOGFILE=dsh-app.log

set APPEXE=
for %%P in (
    "%LOCALAPPDATA%\Perplexity\Comet\Application\comet.exe"
    "%PROGRAMFILES(X86)%\Microsoft\Edge\Application\msedge.exe"
    "%PROGRAMFILES%\Google\Chrome\Application\chrome.exe"
    "%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"
) do (
    if exist %%P set APPEXE=%%~P
)

echo Starting DeepSeek Harness server... > "%LOGFILE%"
start "DeepSeek Harness server" /min cmd /c "pnpm dsh web --no-open >> "%LOGFILE%" 2>&1"

:waitloop
timeout /t 1 /nobreak >nul
findstr /c:"dsh web: http" "%LOGFILE%" >nul 2>nul
if errorlevel 1 goto waitloop

for /f "tokens=3" %%A in ('findstr /c:"dsh web: http" "%LOGFILE%"') do set DSH_URL=%%A

if defined APPEXE (
    start "" "!APPEXE!" --app=%DSH_URL%
) else (
    start "" "%DSH_URL%"
)
exit
