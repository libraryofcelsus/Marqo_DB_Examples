@echo off
:: Check if Git is already installed
where git >nul 2>nul
if %errorlevel% equ 0 (
    echo Git is already installed.
) else (
    :: Download Git installer
    echo Downloading Git installer...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/git-for-windows/git/releases/download/v2.41.0.windows.3/Git-2.41.0.3-64-bit.exe' -OutFile '%TEMP%\GitInstaller.exe'"

    :: Install Git
    echo Installing Git...
    %TEMP%\GitInstaller.exe /SILENT /COMPONENTS="icons,ext\reg\shellhere,assoc,assoc_sh"

    :: Delete the installer
    del %TEMP%\GitInstaller.exe
)

:: Check if FFmpeg is already installed
where ffmpeg >nul 2>nul
if %errorlevel% == 0 (
    echo FFmpeg is already installed.
    goto install_venv
)

:: Request administrative privileges for setting the PATH
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting administrative privileges...
    powershell -Command "Start-Process cmd -ArgumentList '/c %~f0' -Verb runAs"
    goto end
)

:: Create a directory for FFmpeg
echo Creating directory for FFmpeg...
if not exist ffmpeg_install mkdir ffmpeg_install
cd ffmpeg_install

:: Download FFmpeg
echo Downloading FFmpeg...
powershell -Command "Invoke-WebRequest -Uri 'https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-6.0-essentials_build.zip' -OutFile 'ffmpeg.zip'"

:: Unzip FFmpeg
echo Unzipping FFmpeg...
powershell -Command "Expand-Archive -Path 'ffmpeg.zip' -DestinationPath ."

:: Rename and move folder to C:\
echo Moving FFmpeg to C:\...
if exist C:\ffmpeg rmdir /S /Q C:\ffmpeg
move ffmpeg-* C:\ffmpeg

:: Add FFmpeg to system PATH
echo Adding FFmpeg to PATH...
setx /M PATH "%PATH%;C:\ffmpeg\bin"

:: Clean up
cd ..
rmdir /S /Q ffmpeg_install
echo FFmpeg installation complete!

:install_venv
echo Installing virtual environment and dependencies...

:: Enable delayed variable expansion
setlocal enabledelayedexpansion

cd /d "%~dp0"

:: Create a virtual environment
echo Creating virtual environment...
python -m venv "venv"
if %errorlevel% neq 0 (
    echo Failed to create virtual environment.
    goto end
)

:: Install project dependencies
echo Installing project dependencies...
"venv\Scripts\python" -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install requirements.
    goto end
)

echo Virtual environment setup complete.

echo Press any key to exit...
pause >nul
goto :EOF

:end
echo Script ended.
