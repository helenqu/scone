#!/bin/bash

module load python/3.7-anaconda-2019.10
source activate TF2Env36
python /global/homes/h/helenqu/create_heatmaps/data_cuts.py --config_path /global/homes/h/helenqu/create_heatmaps/create_heatmaps_larger_plasticc_config.yml
