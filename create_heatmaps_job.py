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

# if len(failed_procs) == 0:
#     donefile_info = "SUCCESS"
if len(failed_procs) > 0:
    donefile_info = "CREATE HEATMAPS FAILURE\nindices of failed create heatmaps jobs: {}\ncheck out the LC data files or metadata files at those indices in the config yml at {}\nlogs located at create_heatmaps_i.log, i=failed index".format(failed_procs, args.config_path)
    donefile_path = config.get("donefile", os.path.join(config["heatmaps_path"], "done.txt"))
    
    if not os.path.exists(config["heatmaps_path"]):
        os.makedirs(config["heatmaps_path"])
    with open(donefile_path, "w+") as donefile:
        donefile.write(donefile_info)
   
   exit_code = 1
else:
   exit_code = 0

sys.exit(exit_code)
