#!/bin/bash

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



# Check if conda is already installed
if ! command -v conda &> /dev/null
then
    # install conda
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf ~/miniconda3/miniconda.sh

    # initialize conda
    ~/miniconda3/bin/conda init bash
    ~/miniconda3/bin/conda init zsh

    # Ensure conda is initialized in the script
    source ~/miniconda3/etc/profile.d/conda.sh
fi

# Create and activate a new environment
conda create --name llm python=3.12 -y

# Activate the new environment
conda init && conda activate llm

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6

# Install necessary packages
conda activate llm && conda install pytorch torchvision -c pytorch -y
conda activate llm && pip install vllm 
conda activate llm && pip install transformers accelerate python-dotenv tqdm-loggable datasets flash-attn