#!/bin/bash

# MinerU Test Environment Setup Script
echo "🚀 Setting up MinerU test environment..."

# Use the project's existing virtual environment
PROJECT_ROOT="$(dirname "$0")/../.."
VENV_PATH="$PROJECT_ROOT/venv"

if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Project virtual environment not found at $VENV_PATH"
    echo "Please ensure you're running this from the correct directory"
    exit 1
fi

# Activate the project's virtual environment
echo "🔧 Activating project virtual environment..."
source "$VENV_PATH/bin/activate"

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing MinerU and dependencies..."
pip install -r requirements_mineru.txt

# Download MinerU model weights (this will create magic-pdf.json config)
echo "📥 Downloading MinerU model weights..."
python -c "
try:
    import magic_pdf
    print('✅ MinerU installed successfully')
    print('📝 Model weights will be downloaded on first use')
except ImportError as e:
    print(f'❌ Error importing MinerU: {e}')
"

echo "✅ Setup complete!"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run tests:"
echo "  cd tests/mineru_experiment"
echo "  python test_mineru_parser.py"
