#!/bin/bash
echo "============================================================"
echo " Starting SportsVision Multi-Sport Analytics Web Studio"
echo "============================================================"
echo ""

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "[!] Virtual environment not found. Using system python3."
fi

python3 app.py
