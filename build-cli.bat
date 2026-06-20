@echo off
setlocal enabledelayedexpansion

echo =======================================
echo Building Jan CLI (jan-cli)
echo =======================================

:: 1. Force kill any running instances of jan-cli and llama-server to release locks
echo Killing existing jan-cli.exe and llama-server.exe instances...
taskkill /IM jan-cli.exe /F /T >nul 2>&1
taskkill /IM llama-server.exe /F /T >nul 2>&1

:: 2. Initialize MSVC toolchain
echo Initializing Visual Studio compiler environment...
call "D:\VisualStudio\VC\Auxiliary\Build\vcvarsall.bat" x64

:: 3. Run cargo build inside src-tauri folder
echo Building release binary...
cd /d "%~dp0src-tauri"
cargo build --release --features cli --bin jan-cli
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Cargo build failed!
    exit /b %ERRORLEVEL%
)

:: 4. Copy build artifact to resources/bin
echo Copying binary to resources/bin...
if not exist "resources\bin" (
    mkdir "resources\bin"
)
copy /y "target\release\jan-cli.exe" "resources\bin\jan-cli.exe"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to copy compiled binary to target resources directory!
    exit /b %ERRORLEVEL%
)

echo =======================================
echo [SUCCESS] jan-cli built and staged successfully!
echo Binary path: src-tauri\resources\bin\jan-cli.exe
echo =======================================
