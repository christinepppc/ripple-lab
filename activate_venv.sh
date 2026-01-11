#!/bin/bash
# Activate the virtual environment for ripple-lab

source /vol/cortex/cd4/pesaranlab/ripple-lab/.venv/bin/activate

echo "✅ Virtual environment activated!"
echo "Python: $(python --version)"
echo "Location: $(which python)"
echo ""
echo "To deactivate, run: deactivate"

