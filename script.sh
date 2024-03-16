#!/bin/bash

# Loading the required module
module load anaconda/2023a

#SBATCH --gres=gpu:volta:1

# Run the script
python src/main.py "$@"