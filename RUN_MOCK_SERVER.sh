#!/bin/bash

echo "🧪 Starting DCGAN Mock Server"
echo ""

cd backend

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python3 -m venv venv"
    exit 1
fi

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

# Check Pillow
echo "Checking Pillow installation..."
python -c "from PIL import Image; print('✓ Pillow OK')" 2>&1

if [ $? -ne 0 ]; then
    echo "Installing Pillow..."
    pip install pillow
fi

echo ""
echo "="*60
echo "Starting Mock Server on http://localhost:8000"
echo "This simulates GAN training for testing"
echo "Press Ctrl+C to stop"
echo "="*60
echo ""

# Run mock server
python mock_server.py
