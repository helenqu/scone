import os
import yaml
import argparse
import subprocess
from astropy.table import Table
import numpy as np
from collections import Counter
import h5py

# TODO: how to create system-agnostic sbatch header
SBATCH_HEADER = """#!/bin/bash
#SBATCH -C haswell
#SBATCH --qos={qos}
#SBATCH --nodes 1
#SBATCH --ntasks {ntasks}
#SBATCH --cpus-per-task 2
#SBATCH --time=00:05:00
#SBATCH --output={log_path}

module load tensorflow/intel-2.2.0-py37
cd {scone_path}
python create_heatmaps_job.py --config_path {config_path} --start {start} --end {end}"""

LOG_OUTPUT_DIR = os.path.join(os.path.expanduser('~'), "scone_shellscripts")
SBATCH_FILE = os.path.join(LOG_OUTPUT_DIR, "autogen_heatmaps_batchfile_{index}.sh")
PARENT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
SCONE_PATH = os.path.dirname(PARENT_DIR_PATH) #TODO: this directory structure might change

# HELPER FUNCTIONS
def write_config(config, config_path):
    with open(config_path, "w") as f:
        f.write(yaml.dump(config))

def load_config(config_path):
    with open(config_path, "r") as cfgfile:
        config = yaml.load(cfgfile)
    return config

def load_configs(config_path):
    config = load_config(config_path)
    gentype_config = load_config(os.path.join(PARENT_DIR_PATH, "default_gentype_to_typename.yml"))["gentype_to_typename"]
    return config, gentype_config

# count number of objects per sntype
def get_ids_by_sn_name(metadata_paths, sn_type_id_to_name):
    sntype_to_abundance = {}
    for metadata_path in metadata_paths:
        metadata = Table.read(metadata_path, format='fits')
        sntypes = np.unique(metadata['SNTYPE'])
        for sntype in sntypes:
            sn_name = sn_type_id_to_name[sntype]
            current_value = sntype_to_abundance.get(sn_name, np.array([]))
            sntype_to_abundance[sn_name] = np.concatenate((current_value, metadata[metadata['SNTYPE'] == sntype]['SNID'].astype(np.int32)))
    return sntype_to_abundance

# do class balancing
def class_balance(categorical, abundances, ids_by_sn_name, ids_path):
    if categorical: 
        num_to_choose = min(abundances.values())
        ids_to_choose_from = list(ids_by_sn_name.values())
    else: 
        num_Ias = abundances["SNIa"]
        num_non_Ias = sum(abundances.values()) - num_Ias
        num_to_choose = min(num_Ias, num_non_Ias)

        Ia_ids = ids_by_sn_name["SNIa"]
        non_Ia_ids = [id_ for sntype, ids in ids_by_sn_name.items() for id_ in ids if sntype != "SNIa"]
        ids_to_choose_from = [non_Ia_ids, Ia_ids]

    chosen_ids = []
    for ids_list in ids_to_choose_from:
        chosen_ids = np.concatenate((np.random.choice(ids_list, num_to_choose, replace=False), chosen_ids))
    assert len(chosen_ids) == len(np.unique(chosen_ids))

    print(f"writing {len(chosen_ids)} ids ({len(ids_to_choose_from)} types, {num_to_choose} of each) to {ids_path}")
    f = h5py.File(ids_path, "w")
    f.create_dataset("ids", data=chosen_ids, dtype=np.int32)
    f.close()


# autogenerate some parts of config
def autofill_scone_config(config):
    if "input_path" in config and 'metadata_paths' not in config: # write contents of input_path
        config['metadata_paths'] = [f.path for f in os.scandir(config["input_path"]) if "HEAD" in f.name]
        config['lcdata_paths'] = [path.replace("HEAD", "PHOT") for path in config['metadata_paths']]

    sn_type_id_to_name = config.get("sn_type_id_to_name", GENTYPE_CONFIG)
    config["sn_type_id_to_name"] = sn_type_id_to_name

    ids_by_sn_name = get_ids_by_sn_name(config["metadata_paths"], sn_type_id_to_name)
    abundances = {k:len(v) for k, v in ids_by_sn_name.items()}
    print(f"sn abundances by type: {abundances}")
    config['types'] = list(abundances.keys())

    class_balanced = config.get("class_balanced", False)
    categorical = config.get("categorical", False)

    print(f"class balancing {'not' if not class_balanced else ''} applied for {'categorical' if categorical else 'binary'} classification, check 'class_balanced' key if this is not desired")
    if class_balanced: # then write IDs file
        ids_path = f"{config['heatmaps_path']}/ids.hdf5"
        class_balance(categorical, abundances, ids_by_sn_name, ids_path)
        config["ids_path"] = ids_path

    return config

def format_sbatch_file(idx):
    start = idx*NUM_SIMULTANEOUS_JOBS
    end = min(NUM_PATHS, (idx+1)*NUM_SIMULTANEOUS_JOBS)
    ntasks = end - start

    sbatch_setup_dict = {
        "scone_path": SCONE_PATH,
        "config_path": ARGS.config_path,
        "log_path": LOG_PATH,
        "qos": "regular" if ntasks == NUM_SIMULTANEOUS_JOBS else "shared",
        "ntasks": ntasks,
        "index": idx,
        "start": start,
        "end": end
    }
    sbatch_setup = SBATCH_HEADER.format(**sbatch_setup_dict)
    sbatch_file = SBATCH_FILE.format(**{"index": j})
    with open(sbatch_file, "w+") as f:
        f.write(sbatch_setup)
    print("start: {}, end: {}".format(start, end))
    print(f"launching job {idx} from {start} to {end}")

    return sbatch_file

# START MAIN FUNCTION
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='create heatmaps from lightcurve data')
    parser.add_argument('--config_path', type=str, help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
    ARGS = parser.parse_args()

    if not os.path.exists(LOG_OUTPUT_DIR):
        os.makedirs(LOG_OUTPUT_DIR)

    SCONE_CONFIG, GENTYPE_CONFIG = load_configs(ARGS.config_path)

    SCONE_CONFIG = autofill_scone_config(SCONE_CONFIG)
    write_config(SCONE_CONFIG, ARGS.config_path)

    NUM_PATHS = len(SCONE_CONFIG["lcdata_paths"])
    NUM_SIMULTANEOUS_JOBS = 32 # haswell has 32 physical cores
    LOG_PATH = os.path.join(LOG_OUTPUT_DIR, f"CREATE_HEATMAPS__{os.path.basename(ARGS.config_path)}.log")

    print(f"num simultaneous jobs: {NUM_SIMULTANEOUS_JOBS}")
    print(f"num paths: {NUM_PATHS}")
    print(f"logging to {LOG_PATH}")

    for j in range(int(NUM_PATHS/NUM_SIMULTANEOUS_JOBS)+1):
        sbatch_file = format_sbatch_file(j)
        subprocess.run(["sbatch", sbatch_file])

