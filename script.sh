#!/bin/bash
#SBATCH --gres=gpu:volta:2

# Loading the required module
module load anaconda/2023a

# Run the script
python src/main.py "$@"
