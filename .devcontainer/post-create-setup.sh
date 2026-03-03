#!/bin/bash

echo "Running post-create setup script..."
echo ""
echo "____________________________________________________________________________________________"
echo ""

# Check for .containervenv and create it if it doesn't exist
if [ ! -d ".containervenv" ]; then
    echo "Creating .containervenv virtual environment..."
    python -m venv .containervenv --system-site-packages
else
    echo ".containervenv virtual environment already exists. Skipping creation."
fi

# we need to activate the Poetry environment for the user to use it in the terminal
# so edit the .bashrc file to activate the Poetry environment automatically when the terminal is opened
echo "Activating Poetry environment..."
echo "eval \$(poetry env activate)" >> ~/.bashrc
# also for zsh users, we need to add it to the .zshrc file
echo "eval \$(poetry env activate)" >> ~/.zshrc

echo ""
echo "____________________________________________________________________________________________"
echo ""

echo "Installing Python dependencies with Poetry..."
poetry install

echo ""
echo "____________________________________________________________________________________________"
echo ""

# Must happen after poetry install, otherwise pre-commit won't be available
echo "Installing pre-commit hooks..."
poetry run pre-commit install

echo ""
echo "____________________________________________________________________________________________"
echo ""

echo "Installing Node.js dependencies for the web server..."
cd interface && npm install

echo ""
echo "____________________________________________________________________________________________"
echo ""

if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Running 'nvidia-smi' to check GPU access..."
    nvidia-smi
else
    echo "No NVIDIA GPU detected or 'nvidia-smi' not found. Skipping GPU access test."
fi

echo ""
echo "____________________________________________________________________________________________"
echo ""
echo ""
echo "Post-create setup completed successfully!"
echo ""
echo ""
echo "____________________________________________________________________________________________"
