#!/bin/bash
# Start both backend and frontend servers
# This opens two terminal windows (if available) or provides instructions

echo "🚀 Starting Ripple Analysis GUI Servers"
echo ""
echo "You need TWO terminals running:"
echo ""
echo "TERMINAL 1 - Backend:"
echo "  cd packages/backend"
echo "  ./start_server.sh"
echo ""
echo "TERMINAL 2 - Frontend:"
echo "  cd packages/frontend"
echo "  ./start.sh"
echo ""
echo "Then open: http://localhost:3000"
echo ""
echo "Starting backend in background..."
echo ""

# Start backend in background
cd packages/backend
./start_server.sh &
BACKEND_PID=$!

echo "Backend started (PID: $BACKEND_PID)"
echo "Now start the frontend in another terminal:"
echo "  cd packages/frontend && ./start.sh"
echo ""
echo "Press Ctrl+C to stop backend"

# Wait for interrupt
trap "kill $BACKEND_PID 2>/dev/null; exit" INT TERM
wait $BACKEND_PID

