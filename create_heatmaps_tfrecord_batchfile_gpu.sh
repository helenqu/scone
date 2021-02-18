#!/bin/bash
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 20
#SBATCH --time=04:00:00
#SBATCH --array=31,35-36,39,81
srun /global/homes/h/helenqu/create_heatmaps/create_heatmaps_tfrecord_shellscript.sh
