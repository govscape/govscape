#!/bin/bash

# we need to activate the Poetry environment for the user to use it in the terminal
# so edit the .bashrc file to activate the Poetry environment automatically when the terminal is opened
echo "Activating Poetry environment in the terminal..."
echo "eval \$(poetry env activate)" >> ~/.bashrc
# also for zsh users, we need to add it to the .zshrc file
echo "eval \$(poetry env activate)" >> ~/.zshrc

echo "Running post-create setup script..."

echo ""
echo "____________________________________________________________________________________________"
echo ""

echo "Installing Python dependencies with Poetry..."
poetry install

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
echo ""
echo ""
echo "Post-create setup completed successfully!"
echo ""
echo ""
echo ""
echo ""
echo "____________________________________________________________________________________________"
