#!/bin/bash
set -e

cd /app

# Activate virtual environment or create if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Install dependencies
cd repo
pip install -r requirements.txt

# Restart the service
systemctl restart hnrank