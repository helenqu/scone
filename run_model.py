import os
from model_utils import *
import argparse
import yaml
import h5py
import numpy as np
import pandas as pd
import time

start = time.time()
# GET CONFIG PATH
parser = argparse.ArgumentParser(description='set up the SCONE model')
parser.add_argument('--config_path', type=str, help='absolute or relative path to your yml config file, i.e. "/user/files/config.yml"')
parser.add_argument('--mode', type=str, help='train or predict')
args = parser.parse_args()

# LOAD CONFIG
with open(args.config_path, "r") as cfgfile:
    config = yaml.load(cfgfile)

# DEFINE PARAMS
HEATMAPS_PATH = config['heatmaps_path']
OUTPUT_PATH = config['output_path']
BATCH_SIZE = config['batch_size'] if 'batch_size' in config else 32
NUM_EPOCHS = config['num_epochs']
INPUT_SHAPE = (config['num_wavelength_bins'], config['num_mjd_bins'], 2)
CATEGORICAL = config['categorical']
NUM_TYPES = None
if CATEGORICAL:
    # TODO: should i write num types info into a file after create heatmaps? maybe ids file will be large
    ids_file = h5py.File(config['ids_path'], "r")
    types = [x.decode('utf-8').split("_")[0] for x in ids_file["names"]]
    ids_file.close()
    NUM_TYPES = len(np.unique(types))
TRAIN_PROPORTION = config['train_proportion'] if 'train_proportion' in config else 0.8
PREDICT = True if config["mode"] == "predict" else False

# EXECUTE
model = define_and_compile_model(INPUT_SHAPE, CATEGORICAL, NUM_TYPES)
#TODO: maybe split out the test set processing
train_set, val_set, test_set, train_ids, val_ids, test_ids = split_and_retrieve_data(HEATMAPS_PATH, TRAIN_PROPORTION, PREDICT, INPUT_SHAPE, CATEGORICAL, BATCH_SIZE)
model, history = train(model, train_set, val_set, NUM_EPOCHS)
pd.DataFrame(history.history).to_csv(os.path.join(OUTPUT_PATH, "training_history.csv"))

print("######## CLASSIFICATION REPORT ########")
print("classification finished in {:.2f}min".format((time.time() - start) / 60))
print("last training accuracy value: {}".format(history.history["accuracy"][-1]))
print("last validation accuracy value: {}".format(history.history["val_accuracy"][-1]))
if PREDICT:
    acc = test(model, test_set)
    print("test accuracy: {}\n".format(acc))
    preds_dict = get_predictions(model, test_set, test_ids, CATEGORICAL)
else:
    preds_dict = get_predictions(model, train_set, train_ids, CATEGORICAL)

pd.DataFrame(preds_dict).to_csv(os.path.join(OUTPUT_PATH, "predictions.csv"), index=False)
