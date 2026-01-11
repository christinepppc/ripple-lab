# Backend API Server

FastAPI backend server for the Ripple Analysis GUI.

## Starting the Server

### Option 1: Use the helper script (Recommended)
```bash
cd packages/backend
./start_server.sh
```

### Option 2: Manual start
```bash
# Activate virtual environment
source /vol/cortex/cd4/pesaranlab/ripple-lab/.venv/bin/activate

# Change to packages directory
cd /vol/cortex/cd4/pesaranlab/ripple-lab/packages

# Set PYTHONPATH
export PYTHONPATH="/vol/cortex/cd4/pesaranlab/ripple-lab/packages:$PYTHONPATH"

# Start server
uvicorn backend.api_server:app --reload --host 127.0.0.1 --port 8000
```

### Option 3: From project root
```bash
source .venv/bin/activate
cd packages
export PYTHONPATH="$(pwd):$PYTHONPATH"
uvicorn backend.api_server:app --reload --host 127.0.0.1 --port 8000
```

## API Endpoints

Once the server is running, you can access:

- **API Server**: http://localhost:8000
- **Interactive API Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## Endpoints

- `POST /load` - Load LFP data
- `POST /detect` - Detect ripples
- `POST /normalize` - Normalize ripples
- `POST /reject` - Reject ripples
- `GET /events` - Get all events
- `GET /event/{k}` - Get specific event data

## CORS Configuration

The server is configured to accept requests from `http://localhost:3000` (the frontend dev server).

## Troubleshooting

**ModuleNotFoundError: No module named 'backend'**
- Make sure you're running from the `packages` directory
- Ensure PYTHONPATH includes the packages directory
- Use the `start_server.sh` script which handles this automatically

**Port 8000 already in use**
```bash
lsof -ti:8000 | xargs kill -9
```

**Import errors for ripple_core**
- Make sure `ripple_core` is installed in editable mode:
  ```bash
  cd packages/ripple_core
  pip install -e .
  ```

