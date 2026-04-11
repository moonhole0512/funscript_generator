@echo off
setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "VENV_PIP=%VENV_DIR%\Scripts\pip.exe"

echo === Eroscript Maker Launcher ===
echo.

:: ── 가상환경 생성 (없을 때만) ──────────────────────────────
if not exist "%VENV_PYTHON%" (
    echo [*] Virtual environment not found. Creating .venv ...

    :: Python 탐색 (python → python3)
    set "SYS_PYTHON="
    where python >nul 2>&1
    if !ERRORLEVEL! == 0 (
        for /f "delims=" %%i in ('python -c "import sys; print(sys.version_info.major)"') do set "PY_MAJOR=%%i"
        if !PY_MAJOR! GEQ 3 (
            set "SYS_PYTHON=python"
        )
    )
    if not defined SYS_PYTHON (
        where python3 >nul 2>&1
        if !ERRORLEVEL! == 0 (
            set "SYS_PYTHON=python3"
        )
    )
    if not defined SYS_PYTHON (
        echo.
        echo [!] Python 3.9 or higher is required but was not found.
        echo     Please install Python from https://www.python.org/downloads/
        echo     Make sure to check "Add Python to PATH" during installation.
        echo.
        pause
        exit /b 1
    )

    echo [*] Using: !SYS_PYTHON!
    !SYS_PYTHON! -m venv "%VENV_DIR%"
    if !ERRORLEVEL! neq 0 (
        echo [!] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [*] Virtual environment created at: %VENV_DIR%
    echo.
)

:: ── NVIDIA GPU 감지 ──────────────────────────────────────
set "HAS_NVIDIA=0"
nvidia-smi >nul 2>&1
if !ERRORLEVEL! == 0 (
    set "HAS_NVIDIA=1"
    for /f "tokens=*" %%g in ('nvidia-smi --query-gpu=name --format=csv^,noheader 2^>nul') do (
        echo [*] NVIDIA GPU detected: %%g
    )
)

:: ── 의존성 설치 ──────────────────────────────────────────
echo [*] Installing / checking dependencies...
echo     (This may take several minutes on first run)
echo.
"%VENV_PIP%" install -r "%SCRIPT_DIR%requirements.txt" --progress-bar on
if !ERRORLEVEL! neq 0 (
    echo.
    echo [!] Dependency installation failed. Please check the errors above.
    pause
    exit /b 1
)

:: ── CUDA PyTorch 자동 설치 ───────────────────────────────
if "!HAS_NVIDIA!" == "1" (
    "%VENV_PYTHON%" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
    if !ERRORLEVEL! neq 0 (
        echo.
        echo [*] NVIDIA GPU found but PyTorch CUDA is not enabled.
        echo [*] Installing PyTorch with CUDA 12.1 support...
        echo     (Downloading ~2GB -- please wait)
        echo.
        "%VENV_PIP%" install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --progress-bar on
        if !ERRORLEVEL! neq 0 (
            echo.
            echo [!] CUDA PyTorch installation failed. Continuing with CPU mode.
            echo.
        ) else (
            echo.
            echo [*] CUDA PyTorch installed successfully.
        )
    ) else (
        echo [*] PyTorch CUDA: already enabled.
    )
)
echo.
echo [*] Dependencies OK.
echo.

:: ── 프로그램 실행 ─────────────────────────────────────────
echo [*] Starting Eroscript Generator...
echo.
"%VENV_PYTHON%" "%SCRIPT_DIR%ui.py"

if !ERRORLEVEL! neq 0 (
    echo.
    echo [!] Application exited with error code !ERRORLEVEL!.
    pause
)
endlocal
