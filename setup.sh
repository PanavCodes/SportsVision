#!/bin/bash
echo "========================================================"
echo "SportsVision Spatial Analytics Engine - Setup (Linux/Mac)"
echo "========================================================"
echo ""

echo "[1/4] Creating Python Virtual Environment (venv)..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to create virtual environment. Make sure python3-venv is installed."
    exit 1
fi
echo ""

echo "[2/4] Activating Virtual Environment and Upgrading PIP..."
source venv/bin/activate
python3 -m pip install --upgrade pip
echo ""

echo "[3/4] Installing Dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install dependencies."
    exit 1
fi
echo ""

echo "[4/4] Verifying and Downloading Model Weights..."
python3 setup_models.py
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to download or verify models."
    exit 1
fi
echo ""

echo "========================================================"
echo "Setup Complete! "
echo "========================================================"
echo "To run the pipeline, first activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "Process Basketball video:"
echo "  python main_pipeline.py path/to/basketball_video.mp4"
echo ""
echo "Process Cricket video:"
echo "  python main_pipeline.py path/to/cricket_video.mp4"
echo ""
