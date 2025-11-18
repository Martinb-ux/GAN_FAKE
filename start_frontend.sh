#!/bin/bash

echo "Starting DCGAN Frontend..."
cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
fi

# Run dev server
echo "Starting React development server on http://localhost:5173"
npm run dev
