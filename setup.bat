@echo off
echo ========================================================
echo SportsVision Spatial Analytics Engine - Setup (Windows)
echo ========================================================

echo.
echo [1/4] Creating Python Virtual Environment (venv)...
python -m venv venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment. Make sure Python is installed.
    exit /b %errorlevel%
)

echo.
echo [2/4] Activating Virtual Environment and Upgrading PIP...
call venv\Scripts\activate
python -m pip install --upgrade pip

echo.
echo [3/4] Installing Dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    exit /b %errorlevel%
)

echo.
echo [4/4] Verifying and Downloading Model Weights...
python setup_models.py
if %errorlevel% neq 0 (
    echo [ERROR] Failed to download or verify models.
    exit /b %errorlevel%
)

echo.
echo ========================================================
echo Setup Complete! 
echo ========================================================
echo To run the pipeline, first activate the environment:
echo   venv\Scripts\activate
echo.
echo Process Basketball video:
echo   python main_pipeline.py path\to\basketball_video.mp4
echo.
echo Process Cricket video:
echo   python main_pipeline.py path\to\cricket_video.mp4
echo.
pause
