import os
from model_utils import SconeClassifier
import yaml
import argparse
import pandas as pd
import time

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

# AUTOMATIC: RUNS TRAIN/TEST/PREDICT AS SPECIFIED BY CONFIG
preds_dict, history = SconeClassifier(config).run()

print("######## CLASSIFICATION REPORT ########")
if "accuracy" in history:
    print("classification finished in {:.2f}min".format((time.time() - start) / 60))
    print("last training accuracy value: {}".format(history["accuracy"][-1]))
    print("last validation accuracy value: {}".format(history["val_accuracy"][-1]))
if "test_accuracy" in history:
    print("test accuracy value: {}".format(history["test_accuracy"]))

pd.DataFrame(preds_dict).to_csv(os.path.join(config['output_path'], "predictions.csv"), index=False)
