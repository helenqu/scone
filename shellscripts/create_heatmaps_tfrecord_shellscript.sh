#!/bin/bash
# module load tensorflow/intel-2.2.0-py37
conda activate TF2Env36
cd /global/homes/h/helenqu/scone #so i can find the slurm out
python /global/homes/h/helenqu/scone/create_heatmaps_utils.py --config_path $1 --index $2
#$SLURM_ARRAY_TASK_ID
