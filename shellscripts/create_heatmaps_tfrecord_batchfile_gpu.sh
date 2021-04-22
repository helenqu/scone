#!/bin/bash
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 20
#SBATCH --time=01:00:00

module load tensorflow/gpu-2.2.0-py37
srun python run_model.py --config_path config/plasticc_test_calculated_peakmjd.yml
