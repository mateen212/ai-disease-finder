#!/bin/bash
# Installation and Setup Script for Ubuntu/Debian Systems
# For Hybrid Neuro-Symbolic Clinical Decision Support System

echo "========================================================================"
echo "INSTALLATION SCRIPT"
echo "Hybrid Neuro-Symbolic Clinical Decision Support System"
echo "========================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
python3 --version

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

# Check pip
echo "Checking pip..."
if ! command -v pip3 &> /dev/null; then
    echo -e "${YELLOW}Installing pip...${NC}"
    sudo apt-get update
    sudo apt-get install -y python3-pip
fi

# Install system dependencies
echo ""
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3-dev \
    build-essential \
    libopencv-dev \
    python3-opencv

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip3 install --user --upgrade pip setuptools wheel

# Install Python packages
echo ""
echo "Installing Python packages (this may take several minutes)..."
pip3 install --user -r requirements.txt

# Setup Kaggle credentials
echo ""
echo "========================================================================"
echo "KAGGLE API SETUP"
echo "========================================================================"
echo ""

KAGGLE_DIR="$HOME/.kaggle"
DOWNLOADS_KAGGLE="$HOME/Downloads/kaggle.json"

if [ -f "$KAGGLE_DIR/kaggle.json" ]; then
    echo -e "${GREEN}✓ Kaggle credentials already configured${NC}"
elif [ -f "$DOWNLOADS_KAGGLE" ]; then
    echo "Found kaggle.json in Downloads folder"
    mkdir -p "$KAGGLE_DIR"
    cp "$DOWNLOADS_KAGGLE" "$KAGGLE_DIR/kaggle.json"
    chmod 600 "$KAGGLE_DIR/kaggle.json"
    echo -e "${GREEN}✓ Kaggle credentials configured${NC}"
else
    echo -e "${YELLOW}⚠ Kaggle credentials not found${NC}"
    echo ""
    echo "To download datasets from Kaggle:"
    echo "1. Go to: https://www.kaggle.com/settings"
    echo "2. Scroll to 'API' section"
    echo "3. Click 'Create New API Token'"
    echo "4. Save kaggle.json to ~/Downloads/"
    echo "5. Re-run this script OR run:"
    echo "   mkdir -p ~/.kaggle"
    echo "   cp ~/Downloads/kaggle.json ~/.kaggle/"
    echo "   chmod 600 ~/.kaggle/kaggle.json"
fi

# Create directories
echo ""
echo "Creating project directories..."
mkdir -p data models outputs logs examples

# Download datasets
echo ""
echo "========================================================================"
echo "DATASET DOWNLOAD"
echo "========================================================================"
echo ""
echo "Would you like to download datasets now? (y/n)"
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "Downloading datasets (this will take 10-30 minutes)..."
    python3 download_datasets.py
else
    echo "Skipping dataset download. You can run it later with:"
    echo "  python3 download_datasets.py"
fi

# Verify installation
echo ""
echo "========================================================================"
echo "VERIFICATION"
echo "========================================================================"
echo ""
echo "Running system check..."
python3 quickstart.py 2>&1 | tail -20

# Installation complete
echo ""
echo "========================================================================"
echo "INSTALLATION COMPLETE"
echo "========================================================================"
echo ""
echo -e "${GREEN}✓ Installation successful!${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. Download datasets (if not done):"
echo "   ${YELLOW}python3 download_datasets.py${NC}"
echo ""
echo "2. Train the models:"
echo "   ${YELLOW}python3 train.py --train-all${NC}"
echo ""
echo "3. Run demo:"
echo "   ${YELLOW}python3 main.py --demo${NC}"
echo ""
echo "4. Diagnose patients:"
echo "   ${YELLOW}python3 main.py --patient-data examples/dengue_patient.json${NC}"
echo ""
echo "For help:"
echo "   python3 main.py --help"
echo "   python3 train.py --help"
echo ""
echo "========================================================================"
