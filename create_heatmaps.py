import os, sys
import yaml
import argparse
import create_heatmaps_utils
from multiprocessing import Process

parser = argparse.ArgumentParser(description='create heatmaps from lightcurve data')
parser.add_argument('--config_path', type=str, help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
args = parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as cfgfile:
        config = yaml.load(cfgfile)
    return config

# def write_config(config_data, config_path):
#     with open(config_path, "r") as cfgfile:
#         yaml.dump(config_data, cfgfile)

# print(args.config_path)
# print(os.path.abspath(args.config_path))
# if config["input_path"]:
#     config['metadata_paths'] = [f.path for f in os.scandir(config["input_path"]) if "HEAD.csv" in f.name]
#     config['lcdata_paths'] = [path.replace("HEAD", "PHOT") for path in config['metadata_paths']]
#     write_config(config, args.config_path)

# print(args.config_path)
# print(os.path.abspath(args.config_path))
config = load_config(args.config_path)
<<<<<<< HEAD
num_paths = len(config["lcdata_paths"])

procs = []
for i in range(num_paths):
    proc = Process(target=create_heatmaps_utils.run, args=(config, i))
    proc.start()
    procs.append(proc)
for proc in procs:
    proc.join() # wait until procs are done

failed_procs = []
for i, proc in enumerate(procs):
    if proc.exitcode != 0:
        failed_procs.append(i)

if len(failed_procs) == 0:
    donefile_info = "SUCCESS"
    exit_code = 0
else:
    donefile_info = "indices of failed create heatmaps jobs: {}\ncheck out the LC data files or metadata files at those indices in the config yml at {}\nlogs located at create_heatmaps_i.log, i=failed index".format(failed_procs, args.config_path)
    exit_code = 1

with open(config["donefile"], "w+") as donefile:
    donefile.write(donefile_info)

sys.exit(exit_code)

## TODO: find out how to use pippin sbatch header
# WRITE AND SUBMIT SCRIPT
#    print("sbatch")
#    sbatch_header = """#!/bin/bash

#    #SBATCH --partition=broadwl
#    #SBATCH --account=pi-rkessler
#    #SBATCH --job-name=create_heatmaps
#    #SBATCH --output=test_sbatch_heatmaps.log
#    #SBATCH --time=00:10:00
#    #SBATCH --nodes=1
#    #SBATCH --mem-per-cpu=1000
#    #SBATCH --exclusive
#    #SBATCH --ntasks-per-node=1"""

#    if NUM_PATHS > 1:
#        sbatch_header += "\n#SBATCH --array=0-{}".format(NUM_PATHS-1)
#    else:
#        sbatch_header += "\n#SBATCH --array=0"

#    task = """source activate scone_cpu
#    python create_heatmaps_utils.py --config_path {} --index $SLURM_ARRAY_TASK_ID""".format(CONFIG_PATH)

#    slurm = """{sbatch_header}
#    {task}
#    """

#    format_dict = {"sbatch_header": sbatch_header, "task": task}
#    final_slurm = slurm.format(**format_dict)
#    with open("./slurm.job", "w+") as batchfile:
#        batchfile.write(final_slurm)
#    subprocess.run(["sbatch", "slurm.job"])
#else:
=======
if config["input_path"]:
    config['metadata_paths'] = [f.path for f in os.scandir(config["input_path"]) if "HEAD.csv" in f.name]
    config['lcdata_paths'] = [path.replace("HEAD", "PHOT") for path in config['metadata_paths']]
    write_config(config, args.config_path)

NUM_PATHS = len(config["lcdata_paths"])
SCRIPT_PATH = os.path.abspath("create_heatmaps_tfrecord_shellscript.sh")
CONFIG_PATH = os.path.abspath(args.config_path) 

sbatch_header = """#!/bin/bash

#SBATCH --partition=broadwl
#SBATCH --account=pi-rkessler
#SBATCH --job-name=create_heatmaps
#SBATCH --output=test_sbatch_heatmaps.log
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1000
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1"""

if NUM_PATHS > 1:
    sbatch_header += "\n#SBATCH --array=0-{}".format(NUM_PATHS-1)
else:
    sbatch_header += "\n#SBATCH --array=0"

task = """source activate scone_cpu
python create_heatmaps_utils.py --config_path {} --index $SLURM_ARRAY_TASK_ID""".format(CONFIG_PATH)

slurm = """{sbatch_header}
{task}
"""

format_dict = {"sbatch_header": sbatch_header, "task": task}
final_slurm = slurm.format(**format_dict)
with open("./slurm.job", "w+") as batchfile:
    batchfile.write(final_slurm)

subprocess.Popen("sbatch slurm.job".split())
#for i in range(NUM_PATHS):
#    #TODO: determine a good way to estimate time
#    cmd = "srun -C haswell -q regular -N 1 --time 01:00:00 {} {} {} &".format(SCRIPT_PATH, CONFIG_PATH, i)
#    # subprocess.run("module load tensorflow/intel-2.2.0-py37".split(" "))
#    subprocess.Popen(cmd.split(" "))

# SUBMIT SCRIPT WITHOUT WRITING
#if config["input_path"]:
#    config['metadata_paths'] = [f.path for f in os.scandir(config["input_path"]) if "HEAD.csv" in f.name]
#    config['lcdata_paths'] = [path.replace("HEAD", "PHOT") for path in config['metadata_paths']]
#    write_config(config, args.config_path)

#NUM_PATHS = len(config["lcdata_paths"])
#SCRIPT_PATH = os.path.abspath("create_heatmaps_tfrecord_shellscript.sh")
#CONFIG_PATH = os.path.abspath(args.config_path) 

#for i in range(NUM_PATHS):
#    #TODO: determine a good way to estimate time
#    cmd = "srun --partition broadwl -N 1 --time 01:00:00 {} {} {} &".format(SCRIPT_PATH, CONFIG_PATH, i)
#    print(cmd)
#    # subprocess.run("module load tensorflow/intel-2.2.0-py37".split(" "))
#    subprocess.Popen(cmd.split(" "))
