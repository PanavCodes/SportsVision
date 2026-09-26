@echo off
title CourtVision Spatial Analytics Studio
cd /d "%~dp0"

echo ============================================================
echo  Starting CourtVision Spatial Analytics Web Studio
echo ============================================================
echo.

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo [!] Virtual environment not found. Using system python.
)

python app.py
pause
