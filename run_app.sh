#!/bin/bash
echo "============================================================"
echo " Starting SportsVision Multi-Sport Analytics Web Studio"
echo "============================================================"
echo ""

if [ -f "basketball_env/bin/activate" ]; then
    source basketball_env/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "[!] Virtual environment not found. Using system python3."
fi

python3 app.py
