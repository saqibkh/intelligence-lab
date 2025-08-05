#!/bin/bash

# Stop on any error
set -e

# Check if venv folder already exists
if [ -d "venv" ]; then
    echo "🟡 Virtual environment already exists in ./venv"
else
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate the virtual environment
echo "⚙️  Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
# Check for ROCm (AMD GPU)
if [ -f /opt/rocm/.info/version ]; then
    echo "📦 ROCm detected (AMD GPU). Installing ROCm-specific dependencies from requirements-amd.txt..."
    pip install -r requirements-amd.txt
    pip uninstall -y torch torchvision torchaudio
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
    echo "✅ All packages installed."
else
    echo "📦 ROCm not detected. Assuming NVIDIA GPU."
    echo "Installing CUDA-specific dependencies from requirements-nvidia.txt..."
    pip install -r requirements-nvidia.txt
    echo "✅ All packages installed."
fi

echo "✅ Environment setup complete. Virtual environment is active."
echo "💡 To activate it later: source venv/bin/activate"
