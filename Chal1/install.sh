#!/bin/bash
# Installation and Setup Script for AeroEyes Challenge

echo "=================================================="
echo "AeroEyes Challenge - Installation Script"
echo "=================================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "Verifying installation..."
python test_setup.py

echo ""
echo "=================================================="
echo "Installation complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the quick start guide: python test_setup.py"
echo "3. See QUICKSTART.md for usage examples"
echo ""
