#!/bin/bash
#SBATCH -C haswell
#SBATCH --qos=regular
#SBATCH --tasks-per-node=32
#SBATCH -N 1
#SBATCH --time=35:00:00

module load tensorflow/intel-2.2.0-py37
python /global/homes/h/helenqu/scone/create_heatmaps.py --config_path  /global/homes/h/helenqu/scone/config/snoopy_config.yml
