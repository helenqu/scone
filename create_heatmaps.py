import os
import yaml
import argparse
import subprocess

# TODO: give people the option to use sbatch instead of MP? but how to create system-agnostic sbatch header
SBATCH_HEADER = """#!/bin/bash
#SBATCH -C haswell
#SBATCH --qos=regular
#SBATCH -N 1
#SBATCH --cpus-per-task=32
#SBATCH --time=40:00:00

export OMP_PROC_BIND=true
export OMP_PLACES=threads
export OMP_NUM_THREADS=16

module load tensorflow/intel-2.2.0-py37
python {scone_path}/create_heatmaps_job.py --config_path  {config_path} --start {start} --end {end}"""
SBATCH_FILE = "/global/homes/h/helenqu/scone_shellscripts/autogen_heatmaps_batchfile_{index}.sh"

parser = argparse.ArgumentParser(description='create heatmaps from lightcurve data')
parser.add_argument('--config_path', type=str, help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
parser.add_argument('--index', type=int, help='index of single lc file you wish to create heatmaps for within your config file, i.e. 4. if no index is provided, heatmaps will be created for objects in all lc files')
args = parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as cfgfile:
        config = yaml.load(cfgfile)
    return config

config = load_config(args.config_path)
if "input_path" in config:
    config['metadata_paths'] = [f.path for f in os.scandir(config["input_path"]) if "HEAD.csv" in f.name]
    config['lcdata_paths'] = [path.replace("HEAD", "PHOT") for path in config['metadata_paths']]
    with open(args.config_path, "w") as f:
        f.write(yaml.dump(config))

num_paths = len(config["lcdata_paths"])

num_simultaneous_jobs = 16 # haswell has 16 physical cores
print("num simultaneous jobs: {}".format(num_simultaneous_jobs))
print("num paths: {}".format(num_paths))
