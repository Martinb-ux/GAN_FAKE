#!/bin/bash

echo "Starting DCGAN Backend Server..."
cd backend

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Run diagnostics
echo ""
echo "Running diagnostics..."
python diagnose.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Diagnostics failed! Please fix the errors above."
    echo "If you want to try anyway, run: python main.py"
    exit 1
fi

# Run server
echo ""
echo "✓ Diagnostics passed!"
echo "Starting FastAPI server on http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""
python main.py
