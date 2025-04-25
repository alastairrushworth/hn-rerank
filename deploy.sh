#!/bin/bash
set -e

cd /app

# Activate virtual environment or create if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Restart the service
if [ -f /tmp/restart ]; then
    rm /tmp/restart
    echo "Restarting the service..."
    pkill -f "uvicorn app:app"
fi
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000
