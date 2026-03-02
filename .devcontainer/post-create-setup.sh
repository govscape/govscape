#!/bin/bash

echo "Running post-create setup script..."

echo "Installing Python dependencies with Poetry..."
poetry install

echo "Installing Node.js dependencies for the web server..."
cd interface && npm install

echo "Post-create setup completed successfully!"
echo "You can start the web server with 'npm run dev -- --host' from the 'interface' directory."
