#!/bin/bash -l
#SBATCH -o /ptmp/jekwo/2025/logs/synaptag/%j.out
#SBATCH -e /ptmp/jekwo/2025/logs/synaptag/%j.err
#SBATCH -J synaptag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#SBATCH --time=12:00:00

module purge
module load cuda/12.6
module load python-waterboa/2024.06

eval "$(conda shell.bash hook)"
conda activate synaptag

# Run the model training
cd /ptmp/jekwo/2025/synaptag/
python rfplot.py