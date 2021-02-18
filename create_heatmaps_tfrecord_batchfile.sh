#!/bin/bash
#SBATCH -C haswell
#SBATCH --qos=regular
#SBATCH --tasks-per-node=32
#SBATCH -N 1
#SBATCH --time=01:00:00
#SBATCH --array=0,1

srun /global/homes/h/helenqu/create_heatmaps/create_heatmaps_tfrecord_shellscript.sh
