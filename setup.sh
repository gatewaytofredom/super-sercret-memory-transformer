#!/bin/bash
# Setup script for Neural Memory Transformer with uv

set -e

echo "Setting up Neural Memory Transformer..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add uv to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"
    
    echo "uv installed successfully!"
else
    echo "uv is already installed."
fi

# Create virtual environment
echo "Creating virtual environment..."
uv venv

# Activate virtual environment based on OS
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "Detected Windows. Please run: .venv\\Scripts\\activate"
else
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
uv pip sync requirements.txt

# Install project in development mode
echo "Installing project in development mode..."
uv pip install -e .

echo ""
echo "Setup complete! 🎉"
echo ""
echo "To activate the virtual environment, run:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "  .venv\\Scripts\\activate"
else
    echo "  source .venv/bin/activate"
fi
echo ""
echo "Then you can start training with:"
echo "  python train_model.py"