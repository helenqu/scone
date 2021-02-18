import pandas as pd
import numpy as np
import os
import yaml
import argparse

parser = argparse.ArgumentParser(description='create heatmaps from lightcurve data')
parser.add_argument('--config_path', type=str, help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
parser.add_argument('--lc_files', type=str)
parser.add_argument('--metadata_files', type=str)
parser.add_argument('--num_splits', type=int, default=0, help='number of files to split into')
args = parser.parse_args()

with open(args.config_path, "r") as cfgfile:
    config = yaml.load(cfgfile)

for j, file in enumerate(args.lc_files):
    METADATA_PATH = args.metadata_files[j]
    LCDATA_PATH = file
    SN_TYPE_ID_MAP = config["sn_type_id_to_name"]

    metadata = pd.read_csv(METADATA_PATH, compression="gzip") if os.path.splitext(METADATA_PATH)[1] == ".gz" else pd.read_csv(METADATA_PATH)
    lcdata = pd.read_csv(LCDATA_PATH, compression="gzip") if os.path.splitext(LCDATA_PATH)[1] == ".gz" else pd.read_csv(LCDATA_PATH)
    lcdata_ids = np.array(metadata[(metadata.true_target.isin(SN_TYPE_ID_MAP.keys()))&(metadata.ddf_bool == 1)].object_id)
    print("total unique sne: {}".format(len(lcdata_ids)))

    split_lcdata_ids = np.array_split(lcdata_ids, args.num_splits)
    lcdata_rows = 0
    new_metadata_paths = []
    new_lcdata_paths = []
    for i, subarray in enumerate(split_lcdata_ids):
        metadata_split = metadata[metadata['object_id'].isin(subarray)]
        lcdata_split = lcdata[lcdata['object_id'].isin(subarray)]
        lcdata_rows += len(lcdata_split)
        assert len(metadata_split) == len(subarray)
        print("file {} has {} sne with {} lcdata rows".format(i, len(metadata_split), len(lcdata_split)))

        metadata_split.to_csv("{}_{}.csv".format(METADATA_PATH.split(".")[0], i))
        lcdata_split.to_csv("{}_{}.csv".format(LCDATA_PATH.split(".")[0], i))

        new_metadata_paths.append("{}_{}.csv".format(METADATA_PATH.split(".")[0], i)) #extra dashes for writing to config yml
        new_lcdata_paths.append("{}_{}.csv".format(LCDATA_PATH.split(".")[0], i))
    print("total lcdata rows written: {}, out of total: {}".format(lcdata_rows, len(lcdata[lcdata['object_id'].isin(lcdata_ids)])))
    assert lcdata_rows == len(lcdata[lcdata['object_id'].isin(lcdata_ids)])

    config["metadata_paths"] += new_metadata_paths
    config["lcdata_paths"] += new_lcdata_paths
    config["metadata_paths"].remove(METADATA_PATH)
    config["lcdata_paths"].remove(LCDATA_PATH)
    with open(args.config_path, "w") as cfgfile:
        cfgfile.write(yaml.dump(config))
