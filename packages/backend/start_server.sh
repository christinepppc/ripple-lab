#!/bin/bash
# Start the backend API server
# This script ensures the correct Python path is set

# Activate virtual environment
source /vol/cortex/cd4/pesaranlab/ripple-lab/.venv/bin/activate

# Change to packages directory so backend module can be found
cd /vol/cortex/cd4/pesaranlab/ripple-lab/packages

# Add packages to PYTHONPATH
export PYTHONPATH="/vol/cortex/cd4/pesaranlab/ripple-lab/packages:$PYTHONPATH"

# Run uvicorn with the correct module path
echo "Starting backend server..."
echo "Backend will be available at: http://localhost:8000"
echo "API docs at: http://localhost:8000/docs"
echo ""
uvicorn backend.api_server:app --reload --host 127.0.0.1 --port 8000

