#!/bin/bash

# TRACE Project Setup Script
# This script sets up the virtual environment and installs dependencies

set -e  # Exit on error

echo "=========================================="
echo "TRACE Project Setup"
echo "=========================================="

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "Found: $PYTHON_VERSION"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo ""
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install requirements
echo ""
echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

# Install ipykernel for Jupyter notebook support
echo ""
echo "Installing ipykernel for Jupyter notebook support..."
python -m ipykernel install --user --name=venv --display-name "venv (Python $(python --version | cut -d' ' -f2))"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the virtual environment manually, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start Jupyter notebooks, run:"
echo "  jupyter notebook"
echo ""
echo "Or to start JupyterLab, run:"
echo "  jupyter lab"
echo ""

