import os
import yaml
import argparse
import subprocess

parser = argparse.ArgumentParser(description='create heatmaps from lightcurve data')
parser.add_argument('--config_path', type=str, help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
args = parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as cfgfile:
        config = yaml.load(cfgfile)
    return config

def write_config(config_data, config_path):
    with open(config_path, "r") as cfgfile:
        yaml.dump(config_data, cfgfile)

config = load_config(args.config_path)
if config["input_path"]:
    config['metadata_paths'] = [f.path for f in os.scandir(config["input_path"]) if "HEAD.csv" in f.name]
    config['lcdata_paths'] = [path.replace("HEAD", "PHOT") for path in metadata_paths]
    write_config(config, args.config_path)

NUM_PATHS = len(config["lcdata_paths"])
SCRIPT_PATH = os.path.abspath("create_heatmaps_tfrecord_shellscript.sh")
CONFIG_PATH = os.path.abspath(args.config_path) 

for i in range(NUM_PATHS):
    #TODO: determine a good way to estimate time
    cmd = "srun -C haswell -q regular -N 1 --time 01:00:00 {} {} {} &".format(SCRIPT_PATH, CONFIG_PATH, i)
    # subprocess.run("module load tensorflow/intel-2.2.0-py37".split(" "))
    subprocess.Popen(cmd.split(" "))
