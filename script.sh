#!/bin/bash

# Create necessary directories for data storage and caching
mkdir -p data
mkdir -p data/datasets
mkdir -p data/llm_cache

# Create .env file with default configuration
cat > .env <<EOL
# Set to 1 to output logs to stdout instead of a file
STDOUT=1
# Set to 1 to use half precision (bfloat16/float16) to save memory
HALF_PRECISION=1
# Set to 1 to enable test environment with additional logging
TEST_ENV=1
EOL

# Check if conda is already installed
if ! command -v conda &> /dev/null
then
    # Install miniconda
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf ~/miniconda3/miniconda.sh

    # Initialize conda for bash and zsh shells
    ~/miniconda3/bin/conda init bash
    ~/miniconda3/bin/conda init zsh

    # Ensure conda is initialized in the script
    source ~/miniconda3/etc/profile.d/conda.sh
fi

# Check if CUDA is installed
if ! command -v nvcc &> /dev/null
then
    # Install CUDA 12.6
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-6
fi

# Create and activate a new conda environment for LLM work
conda create --name llm python=3.12 -y

# Install necessary packages
conda activate llm && conda install pytorch torchvision -c pytorch -y
conda activate llm && pip install vllm 
conda activate llm && pip install transformers accelerate python-dotenv tqdm-loggable datasets flash-attn