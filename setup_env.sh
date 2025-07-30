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
if [ -f "requirements.txt" ]; then
    echo "📦 Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
    echo "✅ All packages installed."
else
    echo "❌ requirements.txt not found in the current directory."
    deactivate
    exit 1
fi

export PS1="\u@\h:\w\$ "

echo "✅ Environment setup complete. Virtual environment is active."
echo "💡 To activate it later: source venv/bin/activate"

