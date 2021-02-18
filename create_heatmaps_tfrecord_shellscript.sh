#!/bin/bash
module load tensorflow/intel-2.2.0-py37
python /global/homes/h/helenqu/create_heatmaps/create_heatmaps_tfrecord.py --config_path /global/homes/h/helenqu/create_heatmaps/create_heatmaps_config.yml --index $SLURM_ARRAY_TASK_ID
