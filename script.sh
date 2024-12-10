#!/bin/bash

# Check if conda is already installed
if ! command -v conda &> /dev/null
then
    # Install Miniconda if conda is not found
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
    rm Miniconda3-latest-Linux-x86_64.sh

    # Add Miniconda to PATH
    export PATH="$HOME/miniconda/bin:$PATH"

    # Initialize conda for bash and zsh
    conda init
fi

# Create and activate a new environment
conda create --name llm python=3.12 -y

# Activate the new environment
conda init && conda activate llm

# Install necessary packages
conda activate llm && conda install pytorch torchvision -c pytorch -y
conda activate llm && pip install transformers accelerate python-dotenv tqdm-loggable datasets flash-attn

# Create necessary directories
mkdir -p data
mkdir -p data/datasets
mkdir -p data/llm_cache

# Create .env file
cat > .env <<EOL
STDOUT=1
HALF_PRECISION=1
TEST_ENV=1
EOL

