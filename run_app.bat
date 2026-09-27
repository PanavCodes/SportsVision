@echo off
title SportsVision Multi-Sport Spatial Analytics Studio
cd /d "%~dp0"

echo ============================================================
echo  Starting SportsVision Multi-Sport Analytics Web Studio
echo ============================================================
echo.

if exist "basketball_env\Scripts\activate.bat" (
    call basketball_env\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo [!] Virtual environment not found. Using system python.
)

python app.py
pause
