#!/bin/bash
source activate scone_cpu
cd /project2/rkessler/SURVEYS/DES/USERS/helenqu/scone
python /project2/rkessler/SURVEYS/DES/USERS/helenqu/scone/create_heatmaps_utils.py --config_path $1 --index $2
