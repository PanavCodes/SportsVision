#!/bin/bash
echo "========================================================"
echo "CourtVision Spatial Analytics Engine - Setup (Linux/Mac)"
echo "========================================================"
echo ""

echo "[1/3] Creating Python Virtual Environment (venv)..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to create virtual environment. Make sure python3-venv is installed."
    exit 1
fi
echo ""

echo "[2/3] Activating Virtual Environment and Upgrading PIP..."
source venv/bin/activate
python3 -m pip install --upgrade pip
echo ""

echo "[3/3] Installing Dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install dependencies."
    exit 1
fi
echo ""

echo "========================================================"
echo "Setup Complete! "
echo "========================================================"
echo "To run the pipeline, first activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "Then run:"
echo "  python main_pipeline.py data/videos/fiba_first_half.mp4"
echo ""
