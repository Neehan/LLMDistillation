#!/bin/bash

# Skip Layer Distillation for LLMs - Setup Script
echo "===== Setting up Skip Layer Distillation project ====="

# Detect platform
PLATFORM=$(uname)
echo "Detected platform: $PLATFORM"

# Create necessary directories for data storage and caching
echo -e "\n[1/6] Creating necessary directories..."
mkdir -p data
mkdir -p data/datasets
mkdir -p data/llm_cache
mkdir -p data/encodings_train
mkdir -p data/encodings_test
mkdir -p data/logs
echo "Directories created successfully."

# Create .env file with default configuration if it doesn't exist
echo -e "\n[2/6] Setting up environment configuration..."
if [ ! -f .env ]; then
    echo "Creating default .env file..."
    cat > .env <<EOL
# Environment configuration for Skip Layer Distillation

# Logging configuration
# Set to 1 to output logs to stdout instead of a file
STDOUT=1

# Hardware and precision settings
# Set to 1 if CUDA is available (will be auto-detected in code)
HAS_CUDA=0
# Set to 1 to use half precision (bfloat16/float16) to save memory
USE_HALF_PRECISION=1

# Development settings
# Set to 1 to enable test environment with additional logging and limited data
TEST_ENV=0

# Hugging Face authentication
# Add your Hugging Face token here to access gated models
HF_TOKEN=
EOL
    echo ".env file created with default settings."
    echo "NOTE: To use Phi-3 models, you need to add your Hugging Face token to the .env file."
    echo "      Edit the .env file and set HF_TOKEN=your_token_here"
else
    echo ".env file already exists, skipping."
    # Check if HF_TOKEN is set in the .env file
    if grep -q "HF_TOKEN=" .env && ! grep -q "HF_TOKEN=.*[a-zA-Z0-9]" .env; then
        echo "WARNING: HF_TOKEN is not set in your .env file."
        echo "         To use Phi-3 models, you need to add your Hugging Face token."
        echo "         Edit the .env file and set HF_TOKEN=your_token_here"
    fi
fi

# Check if conda is already installed
echo -e "\n[3/6] Setting up Miniconda..."
if ! command -v conda &> /dev/null
then
    echo "Miniconda not found, installing..."
    # Install miniconda
    mkdir -p ~/miniconda3
    
    # Download appropriate Miniconda installer based on platform
    if [ "$PLATFORM" = "Darwin" ]; then
        # macOS
        if [ "$(uname -m)" = "arm64" ]; then
            # Apple Silicon
            wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -O ~/miniconda3/miniconda.sh
        else
            # Intel
            wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ~/miniconda3/miniconda.sh
        fi
    else
        # Linux and others
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    fi
    
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf ~/miniconda3/miniconda.sh

    # Initialize conda for bash and zsh shells
    ~/miniconda3/bin/conda init bash
    ~/miniconda3/bin/conda init zsh

    # Ensure conda is initialized in the script
    source ~/miniconda3/etc/profile.d/conda.sh
    echo "Miniconda installed successfully."
else
    echo "Miniconda already installed, skipping."
    # Ensure conda is initialized in the script
    source $(conda info --base)/etc/profile.d/conda.sh
fi

# Check if CUDA is needed and available (Linux only)
echo -e "\n[4/6] Setting up CUDA..."
if [ "$PLATFORM" = "Linux" ]; then
    if ! command -v nvcc &> /dev/null
    then
        echo "CUDA not found, installing CUDA 12.6..."
        # Install CUDA 12.6
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
        sudo apt-get update
        sudo apt-get -y install cuda-toolkit-12-6
        echo "CUDA 12.6 installed successfully."
        
        # Update .env file to indicate CUDA is available
        sed -i 's/HAS_CUDA=0/HAS_CUDA=1/g' .env
    else
        echo "CUDA already installed, skipping."
        # Update .env file to indicate CUDA is available
        sed -i 's/HAS_CUDA=0/HAS_CUDA=1/g' .env
    fi
else
    echo "CUDA installation skipped (not on Linux)."
fi

# Create and activate a new conda environment for LLM work
echo -e "\n[5/6] Setting up Python environment and dependencies..."
echo "Creating conda environment 'llm' with Python 3.12..."
conda create --name llm python=3.12 -y

# Install necessary packages
echo "Installing required packages..."
conda activate llm

# Install PyTorch with appropriate backend
if [ "$PLATFORM" = "Darwin" ]; then
    # macOS - MPS backend
    echo "Installing PyTorch with MPS backend for macOS..."
    conda install pytorch torchvision -c pytorch -y
else
    # Linux - CUDA backend if available
    echo "Installing PyTorch with CUDA backend..."
    conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
    
    # If on Linux with CUDA, uncomment flash-attn in requirements.txt
    if [ -f .env ] && grep -q "HAS_CUDA=1" .env; then
        echo "Enabling flash-attn for CUDA acceleration..."
        sed -i 's/# flash-attn/flash-attn/g' requirements.txt
    fi
fi

# Install dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# Run data preparation
echo -e "\n[6/6] Preparing training and test data..."
echo "Running data preparation script..."
# Check if data already exists
if [ -f "data/encodings_test/chunk_0.pt" ] && [ -f "data/encodings_train/chunk_0.pt" ]; then
    echo "Training and test data already exist. Skipping data preparation."
    echo "To force reprocessing, run: python -m src.setup --force"
else
    echo "Preparing data. This may take some time..."
    # Ensure STDOUT is set to 1 for visible progress bars
    export STDOUT=1
    
    # Check if we're in test environment
    if [ -f .env ] && grep -q "TEST_ENV=1" .env; then
        echo "Test environment detected. Using smaller dataset..."
        python -m src.setup --train_samples 10000 --test_samples 3000 --train_chunk_size 2000
    else
        # let the file defaults take over in prod
        echo "Using default dataset size..."
        python -m src.setup
    fi
    echo "Data preparation completed."
fi

echo -e "\n===== Setup complete! ====="
echo "You can now train the model with:"
echo "  conda activate llm && python -m src.models.phi_3_skip_layer_distiller" 