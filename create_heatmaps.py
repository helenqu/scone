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
#SBATCH --time=20:00:00
#SBATCH --output={log_path}

export OMP_PROC_BIND=true
export OMP_PLACES=threads
export OMP_NUM_THREADS=16

module load tensorflow/intel-2.2.0-py37
python {scone_path}/create_heatmaps_job.py --config_path  {config_path} --start {start} --end {end}"""

LOG_OUTPUT_PATH = os.path.join(os.path.expanduser('~'), "scone_shellscripts")
SBATCH_FILE = os.path.join(LOG_OUTPUT_PATH, "autogen_heatmaps_batchfile_{index}.sh")

if not os.path.exists(LOG_OUTPUT_PATH):
    os.makedirs(LOG_OUTPUT_PATH)

parser = argparse.ArgumentParser(description='create heatmaps from lightcurve data')
parser.add_argument('--config_path', type=str, help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
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
for j in range(int(num_paths/num_simultaneous_jobs)+1):
    start = j*num_simultaneous_jobs
    end = min(num_paths, (j+1)*num_simultaneous_jobs)

    print("start: {}, end: {}".format(start, end))
    sbatch_setup_dict = {
        "scone_path": os.path.dirname(os.path.abspath(__file__)), # parent directory of this file
        "config_path": args.config_path,
        "log_path": os.path.join(LOG_OUTPUT_PATH, f"CREATE_HEATMAPS__{os.path.basename(args.config_path)}.log"),
        "index": j,
        "start": start,
        "end": end
    }
    sbatch_setup = SBATCH_HEADER.format(**sbatch_setup_dict)
    sbatch_file = SBATCH_FILE.format(**{"index": j})
    with open(sbatch_file, "w+") as f:
        f.write(sbatch_setup)
    print(f"launching job {j} from {start} to {end}")
    subprocess.run(["sbatch", sbatch_file])
