#!/usr/bin/env python
#
# Mar 6 2024 RK 
#  +  minor refactor in main to accept optional --heatmaps_subdir argument that
#     is useful for side-by-side testing of scone codes or options. This code
#     should still be compatible with both original and refactored scone codes
#
import os
import numpy as np
import pandas as pd
import yaml
import tensorflow as tf
from tensorflow.keras import layers, models, utils, optimizers
import h5py
import time
import json
import argparse
import atexit
import time  # 3rd Sept, 2025, A. Mitra - Added for performance timing and progress tracking
import psutil  # 3rd Sept, 2025, A. Mitra - Added for real-time memory usage monitoring

from   data_utils  import *
from   scone_utils import *   # RK - should merge with data_utils ?
import scone_utils as util

# =====================================================
# =====================================================

class SconeClassifier():
    # define my own reshape layer
    class Reshape(layers.Layer):
        def call(self, inputs):
            return tf.transpose(inputs, perm=[0,3,2,1])

        def get_config(self): # for model saving/loading
            return {}

    def __init__(self, config):
        self.scone_config = config  
        self.seed = config.get("seed", 42)
        self.process = psutil.Process()  # 3rd Sept, 2025, A. Mitra - Initialize process object for memory monitoring

        self.output_path    = config['output_path']
        self.heatmaps_paths = config['heatmaps_paths'] if 'heatmaps_paths' in config else config['heatmaps_path'] # #TODO(6/21/23): eventually remove, for backwards compatibility
        self.mode = config["mode"]

        self.strategy = tf.distribute.MirroredStrategy()
        self.batch_size_per_replica = config.get('batch_size', 32)
        self.batch_size = self.batch_size_per_replica * self.strategy.num_replicas_in_sync
        logging.info(f"batch size in config: {self.batch_size_per_replica}, num replicas: {self.strategy.num_replicas_in_sync}, true batch size: {self.batch_size}")

        self.num_epochs = config['num_epochs']
        self.input_shape = (config['num_wavelength_bins'], config['num_mjd_bins'], 2)
        self.categorical = config.setdefault('categorical',False)
        self.types = config.get('types', None)
        if self.categorical and self.types is None:
            raise KeyError('cannot perform categorical classification without knowing the number of source types! please specify the `types` key in your config file to reflect this information')
            # TODO: should i write num types info into a file after create heatmaps? maybe ids file will be large
            # ids_file = h5py.File(config['ids_path'], "r")
            # types = [x.decode('utf-8').split("_")[0] for x in ids_file["names"]]
            # ids_file.close()
            # self.num_types = len(np.unique(types))
        self.num_types = len(self.types) if self.categorical else 2
        self.train_proportion = config.get('train_proportion', 0.8)
        self.with_z = config.get('with_z', False)
        self.abundances = None
        self.train_set = self.val_set = self.test_set = None
        self.class_balanced = config.get('class_balanced', True)
        self.external_trained_model = config.get('trained_model')
        self.prob_column_name = config.setdefault('prob_column_name', "PROB_SCONE") # RK
        self.verbose_data_loading = config.get('verbose_data_loading', False)  # 3rd Sept, 2025, A. Mitra - Enable detailed progress reporting during data loading
        
        # Debug flag system for development/testing  # 3rd Sept, 2025, A. Mitra - Flexible system for switching between implementations
        self.debug_flag = config.get('debug_flag', 0)  # 3rd Sept, 2025, A. Mitra - Allow runtime switching between legacy/refactored code
        self._setup_debug_modes()  # 3rd Sept, 2025, A. Mitra - Configure debug behavior based on flag value

        self.LEGACY = 'sim_fraction' in config
        self.REFAC  = not self.LEGACY
        logging.info(f"LEGACY code: {self.LEGACY}")

        return
    
    def _setup_debug_modes(self):  # 3rd Sept, 2025, A. Mitra - New method to centralize debug flag configuration for easy maintenance
        """Setup debug modes based on debug_flag value.
        
        Debug flag meanings:
        0    = Production mode (default) - uses legacy retrieve_data  # 3rd Sept, 2025, A. Mitra - Changed default to use legacy implementation for stability
        1    = Verbose logging
        901  = Use refactored retrieve_data with basic logging  # 3rd Sept, 2025, A. Mitra - Refactored implementation with enhanced monitoring
        902  = Use refactored retrieve_data with verbose logging  # 3rd Sept, 2025, A. Mitra - Refactored with detailed progress tracking
        1000+ = Reserved for future debug modes
        """
        
        # Define debug mode constants for clarity
        self.DEBUG_MODES = {
            'PRODUCTION': 0,
            'VERBOSE': 1,
            'REFAC_RETRIEVE': 901,
            'REFAC_RETRIEVE_VERBOSE': 902,
        }
        
        # Apply debug settings
        if self.debug_flag == self.DEBUG_MODES['PRODUCTION']:
            logging.info("Debug Mode: Production mode - using LEGACY retrieve_data")  # 3rd Sept, 2025, A. Mitra - Default now uses legacy for stability
        elif self.debug_flag == self.DEBUG_MODES['VERBOSE']:
            self.verbose_data_loading = True
            logging.info("Debug Mode: Verbose logging enabled")
        elif self.debug_flag == self.DEBUG_MODES['REFAC_RETRIEVE']:
            logging.info("Debug Mode: Using REFACTORED retrieve_data")
        elif self.debug_flag == self.DEBUG_MODES['REFAC_RETRIEVE_VERBOSE']:
            self.verbose_data_loading = True
            logging.info("Debug Mode: Using REFACTORED retrieve_data with verbose logging")
        elif self.debug_flag > 0:
            logging.info(f"Debug Mode: Custom debug flag {self.debug_flag}")

    def write_summary_file(self, history):

        # created Mar 2024 by R.Kessler
        # write YAML formatted summary that can be parsed by downstream
        # pipeline components.

        summary_file = os.path.join(self.output_path, SCONE_SUMMARY_FILE)
        logging.info(f"Write formatted summary to {SCONE_SUMMARY_FILE}")

        accuracy_dict = self.get_accuracy_dict(history)
        t_hr = (time.time() - self.t_start)/3600.0

        if self.mode == MODE_TRAIN:
            PROGRAM_CLASS = PROGRAM_CLASS_TRAINING
        elif self.mode == MODE_PREDICT :
            PROGRAM_CLASS = PROGRAM_CLASS_PREDICT
        else:
            PROGRAM_CLASS = "UNKNOWN"

        with open(summary_file,"wt") as s:
            s.write(f"PROGRAM_CLASS:  {PROGRAM_CLASS}\n")
            s.write(f"CPU_SUM:        {t_hr:.2f}  # hr \n")

            s.write(f"ACCURACY:\n")
            for acc_type, acc_value in accuracy_dict.items():
                if isinstance(acc_value,float):
                    acc_value = f"{acc_value:.4f}"
                s.write(f"  - {acc_type}:  {acc_value} \n")            
                
        # - - - - - - - - - - - - -
        # append some heatmap info.
        # scoop up informat from heatmap summary and transfer some of it
        # to training summary        
        heatmap_summ_file = os.path.join(self.heatmaps_paths, SCONE_SUMMARY_FILE)
        if not os.path.exists(heatmap_summ_file): return  # no summary for legacy
        heatmap_summ_info = util.load_config_expandvars(heatmap_summ_file, [] )
        heatmap_dict_copy = {}  # init dict to write into train-summary file
        copy_keys_heatmap = [ 'N_LC', 'INPUT_DATA_DIRS',  'SNID_SELECT_FILES',  'PRESCALE_HEATMAPS' ]

        for key in copy_keys_heatmap:
            if key in heatmap_summ_info:
                tmp_dict = { key : heatmap_summ_info[key] }            
                heatmap_dict_copy.update(tmp_dict)

        with open(summary_file, 'a+') as s:
            s.write(f"\n")
            s.write(f"# info from heatmap creation:\n")
            s.write(f"HEATMAPS_PATH:  {self.heatmaps_paths}   # input to training\n")
            yaml.dump(heatmap_dict_copy, s, default_flow_style=False)

        return

    def get_accuracy_dict(self, history):
        accuracy_dict = {
            'training'   : None,   # vast majority
            'validation' : None,   # small subset
            'test'       : None    # no existing?
        }

        if "accuracy" in history:
            accuracy_dict['training'] = np.round(history["accuracy"][-1], 3)

        if "val_accuracy" in history:
            accuracy_dict['validation'] = history["val_accuracy"][-1]

        if "test_accuracy" in history:
            accuracy_dict['test']       = history["test_accuracy"]
        
        return accuracy_dict

    def write_predict_csv_file(self, predict_dict):

        # Created Apr 4 2024 by R.Kessler
        # write predictions to csv file.

        # create csv file from predictoins
        predict_file = os.path.join(self.output_path, PREDICT_CSV_FILE_BASE)
        pd.DataFrame(predict_dict).to_csv(predict_file, index=False) 

        if self.LEGACY : return

        # re-write csv file with snid as first column, and with prob_preds
        # renamed based on user input key prob_column_name.
        # The code below was moved from Pippin to here so that scone
        # controls all output.        
        predictions = pd.read_csv(predict_file)
        if "pred_labels" in predictions.columns :
            predictions = predictions[["snid", "pred_labels"]] # make sure snid is the first col
            predictions = predictions.rename(columns={"pred_labels": self.prob_column_name })
            predictions.to_csv(predict_file, index=False)
            logging.info(f"SCONE prediction file: {predict_file}")            

        return

    def _print_report_and_save_history(self, history):

        logging.info("######## CLASSIFICATION REPORT ########")
        
        t_minutes = (time.time() - self.t_start) / 60.0
        logging.info(f"classification finished in {t_minutes:.2f} min")

        if "accuracy" in history:
            tmp_accuracy = np.round(history["accuracy"][-1], 3)
            logging.info(f"last training accuracy value: {tmp_accuracy}")

        if "val_accuracy" in history:
            tmp_accuracy = history["val_accuracy"][-1]
            logging.info(f"last validation accuracy value: {tmp_accuracy}")

        if "test_accuracy" in history:
            tmp_accuracy = history["test_accuracy"]
            logging.info(f"test accuracy value: {tmp_accuracy}" )

        history_json_file = os.path.join(self.output_path, "history.json")
        with open(os.path.join(history_json_file), 'w') as outfile:
            json.dump(history, outfile)

    def run(self):
        tf.random.set_seed(self.seed)

        self.t_start = time.time()
        self.trained_model = None
        self.log_memory_usage("Initial startup")  # 3rd Sept, 2025, A. Mitra - Track memory usage at key stages for debugging

        if self.external_trained_model:
            logging.info(f"loading trained model found at {self.external_trained_model}")
            self.trained_model = models.load_model(self.external_trained_model, custom_objects={"Reshape": self.Reshape})

        if self.mode == MODE_TRAIN:
            self.train_set, self.val_set, self.test_set = self._split_and_retrieve_data()
            self.trained_model, history = self.train()
            history = history.history    

        elif self.mode == MODE_PREDICT:
            self.log_memory_usage("Before loading dataset")  # 3rd Sept, 2025, A. Mitra - Monitor memory before data loading
            raw_dataset = self._load_dataset()
            self.log_memory_usage("After loading raw dataset")  # 3rd Sept, 2025, A. Mitra - Monitor memory after raw dataset creation

            
            debug_flag = self.debug_flag
            DEBUG_MODE_LIST = [self.DEBUG_MODES['REFAC_RETRIEVE'], 
                               self.DEBUG_MODES['REFAC_RETRIEVE_VERBOSE'] ]

            legacy_predict = debug_flag is None or debug_flag == self.DEBUG_MODES['PRODUCTION'] 

            # Choose retrieve_data implementation based on debug flag

            if legacy_predict:
                # Default (0) uses legacy implementation
                logging.info("Using LEGACY retrieve_data implementation")
                dataset, size = self._retrieve_data_legacy(raw_dataset)
            elif debug_flag in DEBUG_MODE_LIST:
                # Flags 901 and 902 use refactored implementation
                dataset, size = self._retrieve_data(raw_dataset)  # refactored Sep 2025, A.Mitra
            else:
                sys.exit(f"n ABORT with undefined debug_flag = {debug_flag}")
                # v.xyz
            
            self.log_memory_usage("Finished processing dataset setup")
            
            logging.info(f"Running scone prediction on full dataset of {size} examples")
            predict_dict, acc = self.predict(dataset)
            
            # Note: Due to TensorFlow's lazy evaluation, actual data processing   # 3rd Sept, 2025, A. Mitra - Important note for users about TF behavior
            # happens during model.predict() above, not during dataset creation
            
            self.log_memory_usage("After prediction")  # 3rd Sept, 2025, A. Mitra - Monitor memory usage after prediction completes
            self.write_predict_csv_file(predict_dict)
            history = { "accuracy": [acc] }
        else :
            pass

        logging.info(f"DONE with scone {self.mode}, print report and save history...")
        self._print_report_and_save_history(history)
        self.write_summary_file(history)

        logging.info("ALL DONE with SCONE.")

    # train the model, returns trained model & training log
    # requires:
    #   - model
    #   - train_set, val_set
    #   - NUM_EPOCHS
    #   - batch_size
    def train(self, train_set=None, val_set=None):
        with self.strategy.scope():
            model = self._define_and_compile_model() if not self.external_trained_model else self.trained_model
            logging.info(model.summary())
            train_set = train_set if train_set is not None else self.train_set
            val_set = val_set if val_set is not None else self.val_set

        if not self.class_balanced:
            logging.info("not class balanced, applying class weights")
            class_weights = {k: (self.batch_size / (self.num_types * v)) for k,v in self.abundances.items()}

        train_set = train_set.map(lambda image, label, 
                                  *args: (image, label)).shuffle(100_000).cache().batch(self.batch_size)
        val_set = val_set.map(lambda image, label, 
                              *args: (image, label)).shuffle(10_000).cache().batch(self.batch_size)
        logging.info("starting to train")
        history = model.fit(
            train_set,
            epochs=self.num_epochs,
            validation_data=val_set,
            verbose=1,
            class_weight=class_weights if not self.class_balanced else None)

        outdir_train_model = f"{self.output_path}/trained_model"
        model.save(outdir_train_model)

        # Oct 2024 RK - make sure output model has g+rw permissions
        cmd_chmod = f"chmod -R g+rw {outdir_train_model}"
        os.system(cmd_chmod)

        # Jun 2024 RK - write mean filter wavelengths to ensure these values
        #               are the same in predict mode.
        self.write_filter_wavelengths(outdir_train_model)  # Jun 2024, RK

        return model, history

    def log_memory_usage(self, step_name):  # 3rd Sept, 2025, A. Mitra - New method for real-time memory monitoring throughout processing
        memory_mb = self.process.memory_info().rss / 1024 / 1024  # 3rd Sept, 2025, A. Mitra - Convert bytes to MB for readable output
        logging.info(f"{step_name}: Memory usage: {memory_mb:.1f} MB")  # 3rd Sept, 2025, A. Mitra - Log memory usage with descriptive step name

    def write_filter_wavelengths(self,outdir_train_model):

        # Created Jun 2024 by R.Kessler
        # write central wavelength per band in train_model folder,
        # so that predict mode can pick up the same filter wavelength
        # for creating heatmaps.

        filter_wavelength_file = f"{outdir_train_model}/{FILTER_WAVE_FILE}"
        logging.info(f"Write mean filter_wave to: {filter_wavelength_file}")
        util.load_SIM_README_DOCANA(self.scone_config)
        util.load_SIM_GENFILTER_WAVE(self.scone_config)
        
        band_to_wave = self.scone_config['band_to_wave']

        # write yaml brute force so that filters appear in same order that they were read
        with open(filter_wavelength_file, 'w') as outfile:
            outfile.write(f"# central wavelength vs band used to create heatmaps for training.\n")
            for band, wave in band_to_wave.items():
                outfile.write(f"{band}:  {wave} \n")
            #yaml.dump(band_to_wave, outfile, default_flow_style=False)

        return

    def predict(self, dataset, dataset_ids=None):
        if self.external_trained_model and not self.trained_model:
            self.trained_model = models.load_model(self.external_trained_model, 
                                                   custom_objects={"Reshape": self.Reshape})

        if not self.trained_model:
            raise RuntimeError('model has not been trained! call `train` on the SconeClassifier instance before predict!')

        dataset = dataset.cache() # otherwise the rest of the dataset operations won't return entries in the same order
        dataset_no_ids = dataset.map(lambda image, label, *_: (image, label)).batch(self.batch_size)

        # Set verbosity based on config  # 3rd Sept, 2025, A. Mitra - Allow user to control prediction verbosity
        predict_verbose = 1 if self.verbose_data_loading else 0  # 3rd Sept, 2025, A. Mitra - Show progress bar during prediction if verbose mode enabled
        logging.info(f"Starting prediction on batches (batch_size={self.batch_size})...")  # 3rd Sept, 2025, A. Mitra - Inform user about batch processing start
        predictions = self.trained_model.predict(dataset_no_ids, verbose=predict_verbose)

        if self.categorical:
            predictions = np.argmax(predictions, axis=1) #TODO: is this the best way to return categorical results? doesnt preserve confidence info
        predictions = predictions.flatten()

        true_labels = dataset.map(lambda _, label, *args: label["label"])
        df_dict = {'pred_labels': predictions, 'true_labels': list(true_labels.as_numpy_iterator())}
        ids = dataset.map(lambda _, label, id_: id_["id"])
        df_dict['snid'] = list(ids.as_numpy_iterator())

        prediction_ints = np.round(predictions)
        acc = float(np.count_nonzero((prediction_ints - list(true_labels.as_numpy_iterator())) == 0)) / len(prediction_ints)

        return df_dict, acc

    def test(self, test_set=None):
        if not self.mode == MODE_PREDICT :
            raise RuntimeError('no testing in train mode')
        if not self.trained_model:
            raise RuntimeError('model has not been trained! call `train` on the SconeClassifier instance before test!')
        test_set = test_set or self.test_set
        test_set = test_set.map(lambda image, label, *_: (image, label)).batch(self.batch_size)
        results = self.trained_model.evaluate(test_set)
        return results[1]

    # defines and compiles, then returns model
    # requires:
    #   - INPUT_SHAPE
    #   - CATEGORICAL
    #   - NUM_TYPES
    def _define_and_compile_model(self, metrics=['accuracy']):
        y, x, _ = self.input_shape

        image_input = tf.keras.Input(shape=self.input_shape, name="image")
        # z_input, z_err_input will only be used when doing classification with redshift
        z_input = tf.keras.Input(shape=(1,), name="z")
        z_err_input = tf.keras.Input(shape=(1,), name="z_err")
        inputs = [image_input] if not self.with_z else [image_input, z_input, z_err_input]

        x = layers.ZeroPadding2D(padding=(0,1))(image_input)
        x = layers.Conv2D(y, (y, 3), activation='elu')(x)
        x = self.Reshape()(x)
        x = layers.BatchNormalization()(x)
        x = layers.ZeroPadding2D(padding=(0,1))(x)
        x = layers.Conv2D(y, (y, 3), activation='elu')(x)
        x = self.Reshape()(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.ZeroPadding2D(padding=(0,1))(x)
        x = layers.Conv2D(int(y/2), (int(y/2), 3), activation='elu')(x)
        x = self.Reshape()(x)
        x = layers.BatchNormalization()(x)
        x = layers.ZeroPadding2D(padding=(0,1))(x)
        x = layers.Conv2D(int(y/2), (int(y/2), 3), activation='elu')(x)
        x = self.Reshape()(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Flatten()(x)
        x = layers.Dropout(0.5)(x)
        if self.with_z:
            x = layers.concatenate([x, z_input, z_err_input])
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

        if self.categorical:
            sn_type_pred = layers.Dense(self.num_types, activation='softmax', name="label")(x)
        else:
            sn_type_pred = layers.Dense(1, activation='sigmoid', name="label")(x)

        model = models.Model(inputs=inputs, outputs=[sn_type_pred])
        opt = optimizers.Adam(learning_rate=5e-5)
        loss = 'sparse_categorical_crossentropy' if self.categorical else 'binary_crossentropy'
        logging.info(metrics)
        model.compile(optimizer=opt,
                      loss=loss,
                      metrics=metrics)

        return model

    def _load_dataset(self):
        if type(self.heatmaps_paths) == list:
            filenames = ["{}/{}".format(heatmaps_path, f.name) for heatmaps_path in self.heatmaps_paths for f in os.scandir(heatmaps_path) if "tfrecord" in f.name]
        else:
            filenames = ["{}/{}".format(self.heatmaps_paths, f.name) for f in os.scandir(self.heatmaps_paths) if "tfrecord" in f.name]


        np.random.shuffle(filenames)
        logging.info(f"Found {len(filenames)} heatmap files")
        logging.info(f"First random heatmap file: {filenames[0]}")
        
        # Show first few files for debugging  # 3rd Sept, 2025, A. Mitra - Help users verify correct files are being loaded
        if len(filenames) > 3:
            logging.info(f"Loading files including: {filenames[:3]}")  # 3rd Sept, 2025, A. Mitra - Display sample of files being processed
        
        # Calculate total size of files to be loaded  # 3rd Sept, 2025, A. Mitra - Inform users about expected data volume
        total_size_mb = sum(os.path.getsize(f) for f in filenames) / (1024 * 1024)  # 3rd Sept, 2025, A. Mitra - Convert bytes to MB for readability
        logging.info(f"Total data size to load: {total_size_mb:.1f} MB")  # 3rd Sept, 2025, A. Mitra - Show total data size to help users plan resource usage

        raw_dataset = tf.data.TFRecordDataset(
            filenames,
            num_parallel_reads=80)
        
        logging.info(f"Dataset created with {80} parallel readers")  # 3rd Sept, 2025, A. Mitra - Inform users about parallel reading configuration

        #print(f"\n xxx raw_dataset = {raw_dataset}\n")  # .xyz
        #sys.stdout.flush() 

        return raw_dataset

    def _retrieve_data_legacy(self, raw_dataset):  # 3rd Sept, 2025, A. Mitra - Renamed from original _retrieve_data for clarity
        dataset_size = sum([1 for _ in raw_dataset])  # 3rd Sept, 2025, A. Mitra - Simple counting method (may cause memory issues on large datasets)
        dataset = raw_dataset.map(lambda x: get_images(x, self.input_shape, self.with_z), num_parallel_calls=40)
        # self.types = [0,1] if not self.categorical else range(0, self.num_types)

        return dataset.apply(tf.data.experimental.ignore_errors()), dataset_size


    def _retrieve_data(self, raw_dataset):  # 3rd Sept, 2025, A. Mitra - New memory-efficient implementation with progress monitoring
        # Memory-efficient processing using TensorFlow's built-in optimizations  # 3rd Sept, 2025, A. Mitra - Improved memory management for large datasets
        
        # Track progress during data processing  # 3rd Sept, 2025, A. Mitra - Initialize progress tracking system
        self._chunk_counter = {'count': 0, 'start_time': time.time()}  # 3rd Sept, 2025, A. Mitra - Counter and timer for progress reporting
        
        # Get size first for progress tracking  # 3rd Sept, 2025, A. Mitra - Determine dataset size for progress calculations
        logging.info("Calculating dataset size...")
        dataset_size = tf.data.experimental.cardinality(raw_dataset).numpy()  # 3rd Sept, 2025, A. Mitra - Use TF's efficient cardinality method
        if dataset_size == tf.data.experimental.UNKNOWN_CARDINALITY:
            # Fallback if cardinality is unknown  # 3rd Sept, 2025, A. Mitra - Handle cases where TF can't determine size
            logging.info("Dataset size unknown, counting records...")
            dataset_size = raw_dataset.reduce(0, lambda x, _: x + 1).numpy()  # 3rd Sept, 2025, A. Mitra - Fallback counting method
        
        logging.info(f"Total dataset size: {dataset_size} records")
        self._estimated_total_chunks = dataset_size
        
        # Determine reporting interval based on verbosity and dataset size  # 3rd Sept, 2025, A. Mitra - Adaptive progress reporting frequency
        if self.verbose_data_loading:
            # In verbose mode, report more frequently (at least 20 reports)  # 3rd Sept, 2025, A. Mitra - More frequent updates for detailed monitoring
            report_interval = min(100, max(10, dataset_size // 20)) if dataset_size > 0 else 100  # 3rd Sept, 2025, A. Mitra - Calculate optimal reporting frequency
            logging.info(f"Verbose mode: Progress will be reported every {report_interval} records")  # 3rd Sept, 2025, A. Mitra - Inform user about reporting frequency
        else:
            # Normal mode (at least 10 reports)  # 3rd Sept, 2025, A. Mitra - Less frequent updates for normal operation
            report_interval = min(1000, max(100, dataset_size // 10)) if dataset_size > 0 else 1000  # 3rd Sept, 2025, A. Mitra - Balanced reporting frequency
        
        def process_with_progress(x):  # 3rd Sept, 2025, A. Mitra - Inner function to process data with progress tracking
            self._chunk_counter['count'] += 1  # 3rd Sept, 2025, A. Mitra - Increment record counter
            
            # Only report progress during actual processing, not during setup  # 3rd Sept, 2025, A. Mitra - Avoid confusing progress reports during TF pipeline setup
            # Skip the first record which is just pipeline verification  # 3rd Sept, 2025, A. Mitra - TF processes first record for pipeline verification
            if self._chunk_counter['count'] > 1:
                # Report progress at intervals  # 3rd Sept, 2025, A. Mitra - Regular progress updates
                if self._chunk_counter['count'] % report_interval == 0:
                    elapsed = time.time() - self._chunk_counter['start_time']  # 3rd Sept, 2025, A. Mitra - Calculate processing time
                    rate = self._chunk_counter['count'] / elapsed if elapsed > 0 else 0  # 3rd Sept, 2025, A. Mitra - Calculate processing rate
                    memory_mb = self.process.memory_info().rss / 1024 / 1024  # 3rd Sept, 2025, A. Mitra - Monitor current memory usage
                    progress_pct = (self._chunk_counter['count'] / dataset_size * 100) if dataset_size > 0 else 0  # 3rd Sept, 2025, A. Mitra - Calculate completion percentage
                    
                    logging.info(f"Processing record {self._chunk_counter['count']}/{dataset_size} ({progress_pct:.1f}%) | Rate: {rate:.1f} records/sec | Memory: {memory_mb:.1f} MB")  # 3rd Sept, 2025, A. Mitra - Comprehensive progress report
                    
                    # Verbose mode shows estimated time remaining  # 3rd Sept, 2025, A. Mitra - Additional info for verbose users
                    if self.verbose_data_loading:
                        remaining = dataset_size - self._chunk_counter['count']  # 3rd Sept, 2025, A. Mitra - Calculate remaining records
                        eta = remaining / rate if rate > 0 else 0  # 3rd Sept, 2025, A. Mitra - Estimate completion time
                        logging.info(f"  Estimated time remaining: {eta:.1f}s")  # 3rd Sept, 2025, A. Mitra - Show ETA to user
                
                # Also report at 25%, 50%, 75% milestones  # 3rd Sept, 2025, A. Mitra - Show progress at key completion milestones
                elif dataset_size > 0:
                    progress_pct = self._chunk_counter['count'] / dataset_size * 100  # 3rd Sept, 2025, A. Mitra - Calculate current progress percentage
                    if abs(progress_pct - 25) < 0.5 or abs(progress_pct - 50) < 0.5 or abs(progress_pct - 75) < 0.5:  # 3rd Sept, 2025, A. Mitra - Check if at milestone
                        elapsed = time.time() - self._chunk_counter['start_time']  # 3rd Sept, 2025, A. Mitra - Calculate elapsed time
                        rate = self._chunk_counter['count'] / elapsed if elapsed > 0 else 0  # 3rd Sept, 2025, A. Mitra - Calculate processing rate
                        memory_mb = self.process.memory_info().rss / 1024 / 1024  # 3rd Sept, 2025, A. Mitra - Check memory usage at milestone
                        logging.info(f"Progress: {progress_pct:.0f}% ({self._chunk_counter['count']}/{dataset_size}) | Rate: {rate:.1f} records/sec | Memory: {memory_mb:.1f} MB")  # 3rd Sept, 2025, A. Mitra - Report milestone progress
            
            return get_images(x, self.input_shape, self.with_z)  # 3rd Sept, 2025, A. Mitra - Process the actual data using existing get_images function
        
        # Apply processing with progress tracking  # 3rd Sept, 2025, A. Mitra - Create data processing pipeline with monitoring
        dataset = raw_dataset.map(
            process_with_progress,   # 3rd Sept, 2025, A. Mitra - Use progress-tracking wrapper function
            num_parallel_calls=tf.data.AUTOTUNE  # 3rd Sept, 2025, A. Mitra - Let TF optimize parallelism automatically
        ).ignore_errors()  # 3rd Sept, 2025, A. Mitra - Skip corrupted records gracefully
        
        # Use prefetching for better performance and memory management  # 3rd Sept, 2025, A. Mitra - Overlap I/O with computation
        dataset = dataset.prefetch(tf.data.AUTOTUNE)  # 3rd Sept, 2025, A. Mitra - TF manages prefetch buffer size automatically
        
        # Note: actual counting happens during iteration  # 3rd Sept, 2025, A. Mitra - Important note about TF's lazy evaluation
        logging.info(f"Dataset pipeline created with {tf.data.AUTOTUNE} parallel processing")  # 3rd Sept, 2025, A. Mitra - Inform about automatic optimization
        
        return dataset, dataset_size
    
    
    # TODO: only class balance when desired, only split when desired
    # simpler split and retrieve function using tf dataset filter
    # - always class balances with min(abundances)
    # - splits into train, val, test sets
    def _split_and_retrieve_data(self):
        raw_dataset = self._load_dataset()
        
        # Choose retrieve_data implementation based on debug flag
        if hasattr(self, 'DEBUG_MODES') and self.debug_flag in [self.DEBUG_MODES['REFAC_RETRIEVE'], self.DEBUG_MODES['REFAC_RETRIEVE_VERBOSE']]:
            logging.info("Using REFACTORED retrieve_data implementation for training")  # 3rd Sept, 2025, A. Mitra - Enhanced implementation for testing
            dataset, size = self._retrieve_data(raw_dataset)
        else:
            # Default (0) and any other flag uses legacy  # 3rd Sept, 2025, A. Mitra - Changed default to legacy for stability
            logging.info("Using LEGACY retrieve_data implementation for training")
            dataset, size = self._retrieve_data_legacy(raw_dataset)
        
        dataset = dataset.shuffle(size)

        unique, counts = np.unique(list(dataset.map(lambda image, label, *_: label["label"]).as_numpy_iterator()), return_counts=True)
        logging.info(f"dataset abundances: {dict(zip(unique, counts))}")
        num_per_type = min(counts)

        train_set_size_per_type = int(num_per_type*self.train_proportion)
        val_test_proportion = 1-self.train_proportion
        val_test_set_size_per_type = int(num_per_type*val_test_proportion)

        for i in range(self.num_types):
            filtered = dataset.filter(lambda image, label, *_: label["label"] == i)
            curr_train_set = filtered.take(train_set_size_per_type)
            curr_val_test_set = filtered.skip(train_set_size_per_type).take(val_test_set_size_per_type)
            curr_val_set = curr_val_test_set.take(val_test_set_size_per_type//2)
            curr_test_set = curr_val_test_set.skip(val_test_set_size_per_type//2).take(val_test_set_size_per_type//2)

            if i == 0:
                train_set = curr_train_set
                val_set = curr_val_set if self.mode == MODE_PREDICT else curr_val_test_set
                test_set = curr_test_set
            else:
                train_set = train_set.concatenate(curr_train_set)
                val_set = val_set.concatenate(curr_val_set) if self.mode == MODE_PREDICT else val_set.concatenate(curr_val_test_set)
                test_set = test_set.concatenate(curr_test_set)

        # train_set = dataset.take(train_set_size)
        unique, counts = np.unique(list(train_set.map(lambda image, label, *_: label["label"]).as_numpy_iterator()), return_counts=True)
        logging.info(f"train set abundances: {dict(zip(unique, counts))}")

        unique, counts = np.unique(list(val_set.map(lambda image, label, *_: label["label"]).as_numpy_iterator()), return_counts=True)
        logging.info(f"val set abundances: {dict(zip(unique, counts))}")

        unique, counts = np.unique(list(test_set.map(lambda image, label, *_: label["label"]).as_numpy_iterator()), return_counts=True)
        logging.info(f"test set abundances: {dict(zip(unique, counts))}")

        return train_set, val_set, test_set

    # main data retrieval function
    # requires:
    #   - heatmaps_path
    #   - get_images
    #   - stratified_split
    #   - extract_ids
    def _split_and_retrieve_data_stratified(self):
        dataset, size = self._retrieve_data(self._load_dataset())

        train_set, val_set, test_set, self.abundances = stratified_split(dataset, self.train_proportion, self.types, self.mode == MODE_PREDICT, self.class_balanced)

        train_set = train_set.prefetch(tf.data.experimental.AUTOTUNE).cache()
        val_set = val_set.prefetch(tf.data.experimental.AUTOTUNE).cache()
        test_set = test_set.prefetch(tf.data.experimental.AUTOTUNE).cache() if self.mode == MODE_PREDICT else None

        train_set = train_set.map(lambda image, label, *_: (image, label)).batch(self.batch_size)
        val_set = val_set.map(lambda image, label, *_: (image, label)).batch(self.batch_size)
        test_set = test_set.map(lambda image, label, *_: (image, label)).batch(self.batch_size)

        return train_set, val_set, test_set


class SconeClassifierIaModels(SconeClassifier):
    # define my own reshape layer
    # Apr 16 2024:  obsolete ??
    class Reshape(layers.Layer):
        def call(self, inputs):
            return tf.transpose(inputs, perm=[0,3,2,1])

        def get_config(self): # for model saving/loading
            return {}

    def __init__(self, config):
        super().__init__(config)
        self.external_test_sets = config.get('external_test_sets', None)

    def run(self):
        if not self.trained_model:
            _, history = self.train()
        dataset, dataset_ids = self.test_set if self.predict else self.train_set
        preds_dict = self.predict(dataset, dataset_ids)

        if self.predict:
            test_acc = self.test()
            history.history["test_accuracy"] = test_acc

        if self.external_test_sets:
            for test_set_dir in self.external_test_sets:
                Ia_dataset = tf.data.TFRecordDataset(
                    ["{}/{}".format(test_set_dir, f.name) for f in os.scandir(test_set_dir) if "tfrecord" in f.name],
                    num_parallel_reads=80)

                Ia_dataset = Ia_dataset.take(20_000)
                raw_dataset = Ia_dataset.concatenate(self.non_Ia_dataset)

                dataset = raw_dataset.map(lambda x: get_images(x, self.input_shape), num_parallel_calls=tf.data.experimental.AUTOTUNE)
                dataset = dataset.apply(tf.data.experimental.ignore_errors())
                dataset = dataset.shuffle(20000).cache()
                ood_ids, dataset = extract_ids_from_dataset(dataset)
                test_set = dataset.batch(self.batch_size)

                accuracy = self.test(test_set)
                history.history[os.path.basename(test_set_dir) + "_test_accuracy"] = accuracy

        return preds_dict, history

    def _split_and_retrieve_data(self):
        Ia_dataset, nonIa_dataset = self._load_dataset_separate_Ia_non()
        raw_dataset = Ia_dataset.concatenate(non_Ia_dataset).shuffle(400_000)
        dataset = self._retrieve_data(raw_dataset)

        train_set, val_set, test_set = stratified_split(dataset, self.train_proportion, self.types, self.predict)
        train_set = train_set.prefetch(tf.data.experimental.AUTOTUNE).cache()
        val_set = val_set.prefetch(tf.data.experimental.AUTOTUNE).cache()
        test_set = test_set.prefetch(tf.data.experimental.AUTOTUNE).cache() if self.predict else None

        train_set = train_set.map(lambda image, label, *_: (image, label)).batch(self.batch_size)
        val_set = val_set.map(lambda image, label, *_: (image, label)).batch(self.batch_size)
        test_set = test_set.map(lambda image, label, *_: (image, label)).batch(self.batch_size)
        return train_set, val_set, test_set

    def _load_dataset_separate_Ia_non(self):
        nonIa_loc = [path for path in self.heatmaps_path if "non" in path][0]
        Ia_loc = [path for path in self.heatmaps_path if path != nonIa_loc][0]

        Ia_dataset = tf.data.TFRecordDataset(
	    ["{}/{}".format(Ia_loc, f.name) for f in os.scandir(Ia_loc) if "tfrecord" in f.name],
	    num_parallel_reads=80)
        non_Ia_dataset = tf.data.TFRecordDataset(
            ["{}/{}".format(nonIa_loc, f.name) for f in os.scandir(nonIa_loc) if "tfrecord" in f.name],
            num_parallel_reads=80)

        Ia_dataset = Ia_dataset.shuffle(100_000)
        non_Ia_dataset = non_Ia_dataset.shuffle(100_000)

        return Ia_dataset, non_Ia_dataset


def get_args():

    parser = argparse.ArgumentParser(  # 3rd Sept, 2025, A. Mitra - Enhanced argument parser with detailed help
        description='SCONE (Supernova Classification with Neural Networks) - Train or predict using heatmap data',  # 3rd Sept, 2025, A. Mitra - Descriptive tool description
        formatter_class=argparse.RawDescriptionHelpFormatter,  # 3rd Sept, 2025, A. Mitra - Preserve formatting in help text
        epilog="""  # 3rd Sept, 2025, A. Mitra - Detailed help section with examples
Debug flag values:
  0    Production mode (default) - uses legacy retrieve_data  # 3rd Sept, 2025, A. Mitra - Stable default
  1    Verbose logging
  901  Use refactored retrieve_data  # 3rd Sept, 2025, A. Mitra - Enhanced implementation with monitoring
  902  Use refactored retrieve_data with verbose logging  # 3rd Sept, 2025, A. Mitra - Detailed progress tracking

Examples:
  %(prog)s --config_path config.yaml
  %(prog)s --config_path config.yaml --debug_flag 902
  %(prog)s --config_path config.yaml --heatmaps_subdir custom_heatmaps
        """
    )

    parser.add_argument('--config_path',   # 3rd Sept, 2025, A. Mitra - Enhanced with required flag for clarity
                       type=str, 
                       required=True,  # 3rd Sept, 2025, A. Mitra - Made requirement explicit
                       help='Path to YAML configuration file (required)')  # 3rd Sept, 2025, A. Mitra - Clear help message

    parser.add_argument('--heatmaps_subdir',   # 3rd Sept, 2025, A. Mitra - Enhanced help message
                       type=str, 
                       default=HEATMAPS_SUBDIR_DEFAULT,
                       help=f'Alternative heatmaps subdirectory name (default: {HEATMAPS_SUBDIR_DEFAULT})')  # 3rd Sept, 2025, A. Mitra - Show default value

    parser.add_argument('--debug_flag',   # 3rd Sept, 2025, A. Mitra - New debug flag argument for development
                       type=int, 
                       default=None,  # 3rd Sept, 2025, A. Mitra - None allows config file to take precedence
                       metavar='N',  # 3rd Sept, 2025, A. Mitra - Clear placeholder in help
                       help='Debug flag for development/testing (0=production, 1=verbose, 900-902=implementation testing). Overrides config file.')  # 3rd Sept, 2025, A. Mitra - Comprehensive help message

    args = parser.parse_args()
    return args


# ===============================================
#   MAIN
# ===============================================
if __name__ == "__main__":

    util.setup_logging()

    util.print_job_command()

    args = get_args()

    key_expandvar_list = [ 'output_path', 'trained_model' ]
    scone_config = util.load_config_expandvars(args.config_path, key_expandvar_list )

    # define full path to heatmaps based on subdir
    scone_config['heatmaps_path'] = os.path.join(scone_config['output_path'],args.heatmaps_subdir)

    # Handle debug_flag: command-line overrides config file  # 3rd Sept, 2025, A. Mitra - Implement priority system for debug flag
    if args.debug_flag is not None:
        scone_config['debug_flag'] = args.debug_flag  # 3rd Sept, 2025, A. Mitra - Command-line takes highest priority
        logging.info(f"Debug flag set from command line: {args.debug_flag}")  # 3rd Sept, 2025, A. Mitra - Inform user about source
    elif 'debug_flag' not in scone_config:
        scone_config['debug_flag'] = 0  # Default value  # 3rd Sept, 2025, A. Mitra - Production mode as default
    else:
        logging.info(f"Debug flag from config: {scone_config['debug_flag']}")  # 3rd Sept, 2025, A. Mitra - Show config file value being used

    SconeClassifier(scone_config).run()

    # ==== END MAIN ===
