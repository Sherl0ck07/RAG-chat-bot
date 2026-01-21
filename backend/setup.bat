@echo off
REM ==================================================
REM Veltris Doc-Bot Backend Setup (Windows)
REM Uses Python 3.11
REM ==================================================

echo =========================================
echo Veltris Doc-Bot Backend Setup
echo =========================================
echo.

REM --------------------------------------------------
REM Verify Python 3.11 exists
REM --------------------------------------------------
echo Checking for Python 3.11...
py -3.11 --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo ERROR: Python 3.11 not found!
    echo.
    echo Available Python versions:
    py -0
    echo.
    echo Please install Python 3.11 from https://www.python.org
    pause
    exit /b 1
)

REM Display Python version
py -3.11 --version
echo.

REM --------------------------------------------------
REM Remove existing venv if present
REM --------------------------------------------------
IF EXIST venv (
    echo Removing existing virtual environment...
    rmdir /s /q venv
    echo.
)

REM --------------------------------------------------
REM Create virtual environment using Python 3.11
REM --------------------------------------------------
echo Creating virtual environment with Python 3.11...
py -3.11 -m venv venv
IF ERRORLEVEL 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo Virtual environment created successfully!
echo.

REM --------------------------------------------------
REM Activate virtual environment
REM --------------------------------------------------
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM --------------------------------------------------
REM Verify we're using the correct Python
REM --------------------------------------------------
echo Verifying Python in virtual environment...
python --version
echo.

REM --------------------------------------------------
REM Upgrade pip tooling
REM --------------------------------------------------
echo Upgrading pip, setuptools, wheel...
python -m pip install --upgrade pip setuptools wheel
IF ERRORLEVEL 1 (
    echo WARNING: pip upgrade had issues, continuing anyway...
)
echo.

REM --------------------------------------------------
REM Install dependencies
REM --------------------------------------------------
echo Installing dependencies from requirements.txt...
echo This may take a few minutes...
pip install -r requirements.txt
IF ERRORLEVEL 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo.
echo Dependencies installed successfully!
echo.

REM --------------------------------------------------
REM Create .env file if missing
REM --------------------------------------------------
IF NOT EXIST ".env" (
    echo Creating .env file from template...
    copy .env.example .env >nul
    echo.
    echo ========================================
    echo IMPORTANT: Configure your API key now!
    echo ========================================
    echo.
    echo Edit .env and update these values:
    echo   - NEBIUS_API_KEY=your_api_key_here
    echo   - NEBIUS_BASE_URL=https://api.studio.nebius.ai/v1
    echo   - NEBIUS_MODEL_NAME=gpt-4o-mini
    echo.
) ELSE (
    echo .env file already exists (not overwriting)
    echo.
)

REM --------------------------------------------------
REM Final message
REM --------------------------------------------------
echo =========================================
echo Setup Complete!
echo =========================================
echo.
echo Next steps:
echo.
echo 1. Edit .env and add your NEBIUS_API_KEY:
echo    notepad .env
echo.
echo 2. Run data ingestion (one-time):
echo    run_ingestion.bat
echo    OR manually:
echo    venv\Scripts\activate
echo    python ingestion\ingest.py
echo.
echo 3. Start the API server:
echo    run_server.bat
echo    OR manually:
echo    venv\Scripts\activate
echo    python app\main.py
echo.
echo 4. Test the API (in new terminal):
echo    run_tests.bat
echo.
echo =========================================
echo.
pause