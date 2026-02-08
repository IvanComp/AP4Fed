#!/bin/bash

# Define environment directory
ENV_DIR="env"

# Ensure we are in the directory of the script
cd "$(dirname "$0")"

# Check if env exists
if [ ! -d "$ENV_DIR" ]; then
    echo "Creating virtual environment in '$ENV_DIR'..."
    python3 -m venv $ENV_DIR
    
    # Activate
    source $ENV_DIR/bin/activate
    
    echo "Upgrading pip..."
    pip install --upgrade pip
    
    echo "Installing requirements..."
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        echo "Error: requirements.txt not found!"
        exit 1
    fi
else
    # Just activate
    source $ENV_DIR/bin/activate
fi

# Run the Configurator first
echo "Launching Configurator..."
python configurator.py

# Run the application
echo "Starting LightAP4FED..."
python main.py
