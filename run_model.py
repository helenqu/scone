import os
import multiprocessing as mp
from model_utils import SconeClassifier, SconeClassifierIaModels
import yaml
import argparse
import pandas as pd
import time
import json
import subprocess

# assumes we're using cori gpu
SBATCH_HEADER = """#!/bin/bash
#SBATCH -C gpu
#SBATCH --nodes 1
#SBATCH -G {num_gpus}
#SBATCH -c {num_cpus}
#SBATCH --time=04:00:00
#SBATCH --output={log_path}

module purge && module load cgpu
module load tensorflow/gpu-2.2.0-py37
cd {scone_path}
srun python model_utils.py --config_path {config_path}"""


def load_config(config_path):
    with open(config_path, "r") as cfgfile:
        config = yaml.load(cfgfile)
    return config

def format_sbatch_file(config_path, output_path, num_gpus, num_cpus):
    sbatch_file_path = os.path.join(output_path, "scone_job.sh")
    log_path = os.path.join(output_path, f"SCONE__{os.path.basename(config_path)}.log")
    #TODO: currently scone is the immediate parent dir of this file; this directory structure might change
    scone_path = os.path.dirname(os.path.abspath(__file__)) 

    sbatch_setup_dict = {
        "scone_path": scone_path,
        "config_path": config_path,
        "log_path": log_path,
        "num_gpus": num_gpus, #TODO: add these config options in as cli args?
        "num_cpus": num_cpus
    }
    sbatch_setup = SBATCH_HEADER.format(**sbatch_setup_dict)

    with open(sbatch_file_path, "w+") as f:
        f.write(sbatch_setup)
    print(f"launching scone job logging to {log_path}")

    return sbatch_file_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='set up the SCONE model')
    parser.add_argument('--config_path', type=str, help='absolute or relative path to your yml config file, i.e. "/user/files/config.yml"')
    args = parser.parse_args()

    scone_config = load_config(args.config_path)
    output_dir = scone_config["heatmaps_path"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sbatch_file_path = format_sbatch_file(args.config_path, output_dir, scone_config.get("num_gpus", 1), scone_config.get("num_cpus", 10))
    
    subprocess.run(f"module load esslurm && sbatch {sbatch_file_path}", shell=True)
