import os
import multiprocessing as mp
from model_utils import SconeClassifier, SconeClassifierIaModels
import yaml
import argparse
import pandas as pd
import time
import json

start = time.time()
# GET CONFIG PATH
parser = argparse.ArgumentParser(description='set up the SCONE model')
parser.add_argument('--config_path', type=str, help='absolute or relative path to your yml config file, i.e. "/user/files/config.yml"')
args = parser.parse_args()

with open(args.config_path, "r") as cfgfile:
    config = yaml.load(cfgfile)

# MANUAL: TRAIN + TEST
# model = SconeClassifier(args.config_path)
# _, history = model.train()
# pd.DataFrame(history.history).to_csv(os.path.join(OUTPUT_PATH, "training_history.csv"))
# test_acc = model.test()

# MANUAL: TRAIN + PREDICT
# trained_model, history = model.train()
# test_set, test_ids = model.get_test_set()
# trained_model.predict(test_set, test_ids)

# FOR CROSS-TRAINING/TESTING ON DIFF IA MODELS
# def run(config, i):
#     print(config)
#     _, history = SconeClassifierIaModels(config).run()
#     with open("{}/history_snoopy_{}.json".format("/global/homes/h/helenqu", i), "w") as outfile:
#         json.dump(history.history, outfile)
# procs = []
# for i in range(config.get("num_simultaneous_runs", 1)):
#     proc = mp.Process(target=run, args=(config,i))
#     proc.start()
#     procs.append(proc)
# for proc in procs:
#     proc.join() # wait until procs are done
#     print("procs done")

# AUTOMATIC: RUNS TRAIN/TEST/PREDICT AS SPECIFIED BY CONFIG
preds_dict, history = SconeClassifier(config).run()

print("######## CLASSIFICATION REPORT ########")
if "accuracy" in history:
    print("classification finished in {:.2f}min".format((time.time() - start) / 60))
    print("last training accuracy value: {}".format(history["accuracy"][-1]))
    print("last validation accuracy value: {}".format(history["val_accuracy"][-1]))
if "test_accuracy" in history:
    print("test accuracy value: {}".format(history["test_accuracy"]))

with open('/global/cscratch1/sd/erinhay/plasticc/outputs/training/history.json', 'w') as outfile:
    json.dump(history, outfile)

with open('/global/cscratch1/sd/erinhay/plasticc/outputs/training/preds.json', 'w') as outfile:
    json.dump(preds_dict, outfile)

#OUTPUT_PATH = config["heatmaps_path"]
#f = open("{}/preds_dict.txt".format(OUTPUT_PATH), 'w')
#f.write(str(preds_dict))
#f.close()

