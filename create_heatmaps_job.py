#!/usr/bin/env python

import os, sys
import yaml
import argparse
import subprocess
import multiprocessing as mp
from create_heatmaps.manager import CreateHeatmapsManager

parser = argparse.ArgumentParser(description='create heatmaps from lightcurve data')
parser.add_argument('--config_path', type=str, help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
parser.add_argument('--start', type=int, help='metadata/lcdata files index to start processing at')
parser.add_argument('--end', type=int, help='metadata/lcdata files index to stop processing at')
args = parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as cfgfile:
        config = yaml.load(cfgfile, Loader=yaml.Loader)
    # expand env vars
    config['lcdata_paths'] = [os.path.expandvars(path) for path in config['lcdata_paths']]
    config['metadata_paths'] = [os.path.expandvars(path) for path in config['metadata_paths']]
    config['heatmaps_path'] = os.path.expandvars(config['heatmaps_path'])
    config['output_path'] = os.path.expandvars(config['output_path'])
    config['trained_model'] = os.path.expandvars(config['trained_model']) if 'trained_model' in config else None

    return config

def create_heatmaps(config, index):
    CreateHeatmapsManager().run(config, index)

config = load_config(args.config_path)

procs = []
for i in range(args.start, args.end):
    proc = mp.Process(target=create_heatmaps, args=(config, i))
    proc.start()
    procs.append(proc)
for proc in procs:
    proc.join() # wait until procs are done
    print("procs done")

failed_procs = []
for i, proc in enumerate(procs):
    if proc.exitcode != 0:
        failed_procs.append(i)

if len(failed_procs) == 0:
    donefile_info = "CREATE HEATMAPS SUCCESS"
else:
    logfile_path = config.get("heatmaps_logfile", os.path.join(config["heatmaps_path"], f"create_heatmaps__{os.path.basename(args.config_path).split('.')[0]}.log"))
    with open(logfile_path, "a+") as logfile:
        logfile.write("\nindices of failed create heatmaps jobs: {failed_procs}\ncheck out the LC data files or metadata files at those indices in the config yml at {args.config_path}\nsee above for logs")

    donefile_info = f"CREATE HEATMAPS FAILURE"

donefile_path = config.get("heatmaps_donefile", os.path.join(config["heatmaps_path"], "done.txt"))
with open(donefile_path, "w+") as donefile:
    donefile.write(donefile_info)
