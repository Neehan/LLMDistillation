#!/bin/bash
#SBATCH --gres=gpu:volta:1

# Loading the required module
module load anaconda/Python-ML-2023b

# Run the script
python src/main.py "$@"
