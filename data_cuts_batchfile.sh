#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=haswell
#SBATCH --tasks-per-node=32
#SBATCH --time=06:00:00

srun /global/homes/h/helenqu/create_heatmaps/data_cuts_shellscript.sh
