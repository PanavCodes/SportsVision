@echo off
echo ========================================================
echo CourtVision Spatial Analytics Engine - Setup (Windows)
echo ========================================================

echo.
echo [1/3] Creating Python Virtual Environment (venv)...
python -m venv venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment. Make sure Python is installed.
    exit /b %errorlevel%
)

echo.
echo [2/3] Activating Virtual Environment and Upgrading PIP...
call venv\Scripts\activate
python -m pip install --upgrade pip

echo.
echo [3/3] Installing Dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    exit /b %errorlevel%
)

echo.
echo [4/4] Downloading Model Weights...
python setup_models.py
if %errorlevel% neq 0 (
    echo [ERROR] Failed to download models.
    exit /b %errorlevel%
)

echo.
echo ========================================================
echo Setup Complete! 
echo ========================================================
echo To run the pipeline, first activate the environment:
echo   venv\Scripts\activate
echo.
echo Then run:
echo   python main_pipeline.py data\videos\fiba_first_half.mp4
echo.
pause
