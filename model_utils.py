#!/usr/bin/env python
#
# Mar 6 2024 RK 
#  +  minor refactor in main to accept optional --heatmaps_subdir argument that
#     is useful for side-by-side testing of scone codes or options. This code
#     should still be compatible with both original and refactored scone codes
#
# Sep 12 2025: remove self.LEGACY and self.REFAC .. keep only REFAC code 

import os
import sys
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
        
        # Memory optimization settings  # 5th Sept, 2025, A. Mitra - Configure TF for memory efficiency
        self._configure_tf_memory()  # 5th Sept, 2025, A. Mitra - Apply memory optimization settings

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
        self.train_proportion  = config.get('train_proportion', 0.8)
        self.with_z            = config.get('with_z', False)
        self.abundances        = None
        self.train_set         = self.val_set = self.test_set = None
        self.class_balanced    = config.get('class_balanced', True)
        self.external_trained_model = config.get('trained_model')
        self.prob_column_name  = config.setdefault('prob_column_name', "PROB_SCONE") # RK
        
        # Sep 2025 A.Mitra: check memory optimization flags
        self.verbose_data_loading = config.get('verbose_data_loading', False) 
        self.memory_optimize       = config.get('memory_optimize', True) 
        self.streaming_threshold   = config.get('streaming_threshold', 10000) 
        self.force_streaming       = config.get('force_streaming', False)  
        self.gc_frequency = config.get('gc_frequency', 50) 
        self.enable_dynamic_batch_size = config.get('enable_dynamic_batch_size', True)  
        self.enable_micro_batching = config.get('enable_micro_batching', True)  
        self.micro_batch_size      = config.get('micro_batch_size', 4)  
        self.enable_model_quantization = config.get('enable_model_quantization', False) 
        self.quantization_method   = config.get('quantization_method', 'dynamic')  
        self.enable_disk_caching   = config.get('enable_disk_caching', False)  
        self.disk_cache_dir        = config.get('disk_cache_dir', '/tmp/scone_cache') 
        self.ultra_low_memory_mode = config.get('ultra_low_memory_mode', False) 
        self.memory_target_gb      = config.get('memory_target_gb', 50) 
        self.dry_run_mode          = config.get('dry_run_mode', False) 
        self.pause_duration        = config.get('pause_duration', 30)  
        
        self.debug_flag = config.get('debug_flag', 0)  
        self._setup_debug_modes()  

        return
    
    def _configure_tf_memory(self):  
        """
        Created Sep 5 2205 by A.Mitra
        Configure TensorFlow memory settings for large dataset processing.
        """
        # Enable memory growth for GPUs to avoid pre-allocating all memory  
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True) 
                logging.info(f"Configured memory growth for {len(gpus)} GPU(s)")
            except RuntimeError as e:
                logging.warning(f"Failed to configure GPU memory growth: {e}")
        
        # Set mixed precision for memory efficiency (if supported) 
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logging.info("Enabled mixed precision for memory efficiency")
        except Exception as e:
            logging.info(f"Mixed precision not enabled: {e}")
        
        # Configure dataset options for memory efficiency  
        tf.config.threading.set_intra_op_parallelism_threads(0)  
        tf.config.threading.set_inter_op_parallelism_threads(0)  
    
    def _setup_debug_modes(self):  
        """
        Created Sep 2025 by A.Mitra
        Setup debug modes based on debug_flag value.
        
        Debug flag meanings:
        0    = Production mode (default) - uses legacy retrieve_data 
        1    = Verbose logging
        901  = Use refactored retrieve_data with basic logging  
        902  = Use refactored retrieve_data with verbose logging 
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
            logging.info("Debug Mode: Production mode - using LEGACY retrieve_data") 
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


        return
        # end _setup_debug_modes

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

        # Handle dry run mode  # 8th Sept, 2025, A. Mitra - Test baseline memory without data
        if self.dry_run_mode:
            baseline_memory = self._dry_run_memory_baseline()
            logging.info(f"🧪 DRY RUN COMPLETED: Baseline memory usage is {baseline_memory:.2f} GB")
            return  # 8th Sept, 2025, A. Mitra - Exit after dry run

        if self.external_trained_model:
            logging.info(f"loading trained model found at {self.external_trained_model}")
            self.trained_model = models.load_model(self.external_trained_model, custom_objects={"Reshape": self.Reshape})
            self._debug_pause_with_memory_report("Model loaded from disk")  # 8th Sept, 2025, A. Mitra - Pause after model loading

        if self.mode == MODE_TRAIN:
            self.train_set, self.val_set, self.test_set = self._split_and_retrieve_data()
            self._debug_pause_with_memory_report("Training data loaded")  # 8th Sept, 2025, A. Mitra - Pause after data loading
            self.trained_model, history = self.train()
            history = history.history    

        elif self.mode == MODE_PREDICT:
            self.log_memory_usage("Before loading dataset")  # 3rd Sept, 2025, A. Mitra - Monitor memory before data loading
            raw_dataset = self._load_dataset()
            self.log_memory_usage("After loading raw dataset")  # 3rd Sept, 2025, A. Mitra - Monitor memory after raw dataset creation
            self._debug_pause_with_memory_report("Raw dataset loaded")  # 8th Sept, 2025, A. Mitra - Pause after raw data loading

            
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
            self._debug_pause_with_memory_report("Dataset processing completed")  # 8th Sept, 2025, A. Mitra - Pause after data processing
            
            logging.info(f"Running scone prediction on full dataset of {size} examples")
            predict_dict, acc = self.predict(dataset)
            
            # Note: Due to TensorFlow's lazy evaluation, actual data processing   
            # 3rd Sept, 2025, A. Mitra - Important note for users about TF behavior
            # happens during model.predict() above, not during dataset creation
            self.log_memory_usage("After prediction")  
            self._debug_pause_with_memory_report("Prediction completed") 
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
        memory_info = self.process.memory_info()
        system_memory = psutil.virtual_memory()
        
        process_rss_gb = memory_info.rss / (1024**3)
        system_used_gb = system_memory.used / (1024**3)
        system_available_gb = system_memory.available / (1024**3)
        system_total_gb = system_memory.total / (1024**3)
        
        logging.info(f"{step_name}: Process Memory: {process_rss_gb:.2f} GB | System Used: {system_used_gb:.1f}/{system_total_gb:.1f} GB ({system_memory.percent:.1f}%) | Available: {system_available_gb:.1f} GB")

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
            raise RuntimeError('model has not been trained! " \
            "call `train` on the SconeClassifier instance before predict!')

        # Monitor and adjust memory settings if needed 
        self._monitor_and_adjust_memory_settings()
        
        # Apply model quantization for inference if enabled
        if self.enable_model_quantization and self.mode == MODE_PREDICT:
            self.trained_model = self._apply_model_quantization(self.trained_model)

        return self._predict_with_memory_optimization(dataset) 
    
    def _monitor_and_adjust_memory_settings(self): 
        """
        Created Sep 2025 by A. Mitra
        Monitor memory usage and automatically escalate optimization strategies.
        """
        try:
            current_memory_gb = psutil.Process().memory_info().rss / (1024**3)
            memory            = psutil.virtual_memory()
            memory_usage_pct  = (memory.used / memory.total) * 100
            available_gb      = memory.available / (1024**3)
            
            # Log current memory state
            logging.info(f"Memory monitoring: Current usage: {current_memory_gb:.1f}GB, " \
                         f"System: {memory_usage_pct:.1f}%, Available: {available_gb:.1f}GB")
            
            # Auto-escalate optimizations based on memory pressure 
            escalated = False
            
            if current_memory_gb > self.memory_target_gb or memory_usage_pct > 85:
                if not self.ultra_low_memory_mode:
                    logging.warning(f"Memory target exceeded " \
                                    f"({current_memory_gb:.1f}GB > {self.memory_target_gb}GB), enabling ultra-low memory mode")
                    self.ultra_low_memory_mode = True
                    escalated = True
                
                if not self.enable_model_quantization:
                    logging.warning("Critical memory pressure, enabling model quantization")
                    self.enable_model_quantization = True
                    self.quantization_method = "dynamic"
                    escalated = True
                
                if not self.force_streaming:
                    logging.warning("Critical memory pressure, forcing streaming prediction")
                    self.force_streaming = True
                    self.streaming_threshold = min(1000, self.streaming_threshold)
                    escalated = True
                
                # Reduce batch sizes aggressively  # 8th Sept, 2025,
                if current_memory_gb > self.memory_target_gb * 1.2:  #  20% over target
                    self.batch_size = max(1, self.batch_size // 8)
                    self.micro_batch_size = 1
                    logging.warning(f"Emergency memory mode: reducing batch size to " \
                                    f"{self.batch_size}, micro-batch to 1")
                    escalated = True
                    
                    # Force immediate garbage collection 
                    import gc
                    gc.collect()
                    
                    # Check if memory reduced after cleanup  
                    new_memory_gb = psutil.Process().memory_info().rss / (1024**3)
                    memory_freed = current_memory_gb - new_memory_gb
                    logging.info(f"Emergency cleanup freed {memory_freed:.1f}GB, " \
                                 f"new usage: {new_memory_gb:.1f}GB")
            
            if escalated:
                logging.info("Memory optimization settings auto-escalated due to memory pressure")
                self.log_memory_usage("After automatic optimization escalation")
            
            return escalated
            
        except Exception as e:
            logging.error(f"Memory monitoring failed: {e}")
            return False
    
    def _apply_model_quantization(self, model):  # 8th Sept, 2025, A. Mitra - Quantize model for reduced memory usage during inference
        """
        Apply quantization to the model for memory-efficient inference.
        """
        try:
            import tensorflow as tf
            
            if self.quantization_method == 'dynamic':
                # Dynamic range quantization - good balance of speed and accuracy  # 8th Sept, 2025, A. Mitra - Most practical quantization method
                logging.info("Applying dynamic range quantization to model...")
                
                # Convert to TensorFlow Lite with dynamic quantization  # 8th Sept, 2025, A. Mitra - Use TFLite for memory efficiency
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]  # 8th Sept, 2025, A. Mitra - Enable dynamic quantization
                tflite_model = converter.convert()
                
                # Create TFLite interpreter  # 8th Sept, 2025, A. Mitra - Wrapper for quantized model
                interpreter = tf.lite.Interpreter(model_content=tflite_model)
                interpreter.allocate_tensors()
                
                # Create a wrapper class to make it compatible with Keras model interface  # 8th Sept, 2025, A. Mitra - Seamless integration
                class TFLiteModelWrapper:
                    def __init__(self, interpreter):
                        self.interpreter = interpreter
                        self.input_details = interpreter.get_input_details()
                        self.output_details = interpreter.get_output_details()
                    
                    def predict(self, x, verbose=0):
                        # Handle batch prediction  # 8th Sept, 2025, A. Mitra - Process batches efficiently
                        if len(x.shape) == 4:  # Batch of samples
                            results = []
                            for sample in x:
                                self.interpreter.set_tensor(self.input_details[0]['index'], np.expand_dims(sample, 0).astype(np.float32))
                                self.interpreter.invoke()
                                output = self.interpreter.get_tensor(self.output_details[0]['index'])
                                results.append(output[0])
                            return np.array(results)
                        else:  # Single sample
                            self.interpreter.set_tensor(self.input_details[0]['index'], np.expand_dims(x, 0).astype(np.float32))
                            self.interpreter.invoke()
                            return self.interpreter.get_tensor(self.output_details[0]['index'])
                
                quantized_model = TFLiteModelWrapper(interpreter)
                memory_reduction = (model.count_params() * 4 - len(tflite_model)) / (model.count_params() * 4) * 100
                logging.info(f"Dynamic quantization applied. Estimated memory reduction: {memory_reduction:.1f}%")
                return quantized_model
                
            elif self.quantization_method == 'float16':
                # Convert weights to float16 for 50% memory reduction  # 8th Sept, 2025, A. Mitra - Simple half-precision conversion
                logging.info("Converting model to float16 precision...")
                
                # Set mixed precision policy  # 8th Sept, 2025, A. Mitra - Use TensorFlow's mixed precision
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                
                # Clone model with float16 weights  # 8th Sept, 2025, A. Mitra - Convert existing model
                model_config = model.get_config()
                float16_model = tf.keras.Model.from_config(model_config)
                float16_model.set_weights(model.get_weights())
                
                logging.info("Float16 conversion applied. Memory reduction: ~50%")
                return float16_model
            
            else:
                logging.warning(f"Unsupported quantization method: {self.quantization_method}, using original model")
                return model
                
        except Exception as e:
            logging.error(f"Model quantization failed: {e}, using original model")
            return model
    
    def _calculate_intelligent_threshold(self, dataset_size):  # 8th Sept, 2025, A. Mitra - Calculate adaptive streaming threshold
        """
        Calculate an intelligent streaming threshold based on available memory and dataset characteristics.
        """
        base_threshold = self.streaming_threshold
        
        try:
            # Get available memory in MB  # 8th Sept, 2025, A. Mitra - Check system resources
            memory = psutil.virtual_memory()
            available_memory_mb = memory.available / 1024 / 1024
            total_memory_mb = memory.total / 1024 / 1024
            memory_usage_pct = (memory.used / memory.total) * 100
            
            # Estimate memory per sample (rough calculation based on input shape)  # 8th Sept, 2025, A. Mitra - Memory requirement estimation
            # Each sample: (num_wavelength_bins, num_mjd_bins, 2) * 4 bytes (float32) + metadata overhead
            bytes_per_sample = self.input_shape[0] * self.input_shape[1] * self.input_shape[2] * 4 * 2  # 8th Sept, 2025, A. Mitra - Factor of 2 for TF overhead
            mb_per_sample = bytes_per_sample / (1024 * 1024)
            
            # Calculate how many samples could fit in available memory (with safety margin)  # 8th Sept, 2025, A. Mitra - Conservative memory planning
            safety_factor = 0.3  # 8th Sept, 2025, A. Mitra - Use only 30% of available memory for data
            memory_based_threshold = int((available_memory_mb * safety_factor) / mb_per_sample)
            
            logging.info(f"Memory analysis: Available: {available_memory_mb:.1f}MB, Usage: {memory_usage_pct:.1f}%, Estimated {mb_per_sample:.2f}MB per sample")
            
            # Adjust threshold based on memory pressure  # 8th Sept, 2025, A. Mitra - Adaptive threshold based on system state
            if memory_usage_pct > 80:  # 8th Sept, 2025, A. Mitra - High memory pressure
                actual_threshold = min(base_threshold // 2, memory_based_threshold)
                logging.info(f"High memory pressure detected, reducing threshold to {actual_threshold}")
            elif memory_usage_pct > 60:  # 8th Sept, 2025, A. Mitra - Moderate memory pressure
                actual_threshold = min(int(base_threshold * 0.75), memory_based_threshold) 
                logging.info(f"Moderate memory pressure, adjusting threshold to {actual_threshold}")
            else:  # 8th Sept, 2025, A. Mitra - Normal memory conditions
                # Use the larger of configured threshold or memory-based calculation  # 8th Sept, 2025, A. Mitra - Take advantage of available memory
                actual_threshold = max(base_threshold, min(memory_based_threshold, base_threshold * 2))
                if actual_threshold != base_threshold:
                    logging.info(f"Memory available, adjusting threshold from {base_threshold} to {actual_threshold}")
            
            return max(actual_threshold, 1000)  # 8th Sept, 2025, A. Mitra - Minimum reasonable threshold
            
        except Exception as e:
            logging.warning(f"Could not calculate intelligent threshold: {e}, using configured threshold: {base_threshold}")
            return base_threshold
    
    def _calculate_adaptive_batch_size(self, base_batch_size):  # 8th Sept, 2025, A. Mitra - Dynamic batch size based on memory pressure
        """
        Calculate an adaptive batch size based on current memory usage and available memory.
        Enhanced for ultra-low memory targets.
        """
        if not self.enable_dynamic_batch_size:
            return base_batch_size
        
        try:
            memory = psutil.virtual_memory()
            available_memory_mb = memory.available / 1024 / 1024
            total_memory_gb = memory.total / (1024 ** 3)
            current_usage_gb = memory.used / (1024 ** 3)
            memory_usage_pct = (memory.used / memory.total) * 100
            
            # Calculate memory per sample for batch sizing  # 8th Sept, 2025, A. Mitra - Estimate memory requirements
            bytes_per_sample = self.input_shape[0] * self.input_shape[1] * self.input_shape[2] * 4 * 3  # 8th Sept, 2025, A. Mitra - Factor of 3 for gradients + activations
            mb_per_sample = bytes_per_sample / (1024 * 1024)
            
            # Ultra-aggressive memory targeting  # 8th Sept, 2025, A. Mitra - Target specific memory usage
            if self.ultra_low_memory_mode or current_usage_gb > self.memory_target_gb:
                # Calculate batch size to stay within target memory  # 8th Sept, 2025, A. Mitra - Precise memory targeting
                target_memory_mb = self.memory_target_gb * 1024
                current_memory_mb = current_usage_gb * 1024
                
                if current_memory_mb > target_memory_mb:
                    # Already over target, use minimum batch size  # 8th Sept, 2025, A. Mitra - Emergency mode
                    adaptive_batch_size = 1
                    self.micro_batch_size = 1  # 8th Sept, 2025, A. Mitra - Force single-sample micro-batches
                    logging.warning(f"Memory usage {current_usage_gb:.1f}GB exceeds target {self.memory_target_gb}GB, using minimum batch size: 1")
                else:
                    # Calculate batch size to approach but not exceed target  # 8th Sept, 2025, A. Mitra - Conservative approach
                    remaining_memory_mb = target_memory_mb - current_memory_mb
                    max_batch_samples = max(1, int((remaining_memory_mb * 0.3) / mb_per_sample))  # 8th Sept, 2025, A. Mitra - Use 30% of remaining
                    adaptive_batch_size = min(base_batch_size, max_batch_samples)
                    logging.info(f"Ultra-low memory mode: target {self.memory_target_gb}GB, current {current_usage_gb:.1f}GB, batch size: {adaptive_batch_size}")
                
                return adaptive_batch_size
            
            # Standard aggressive batch size reduction under memory pressure  # 8th Sept, 2025, A. Mitra - Original logic enhanced
            if memory_usage_pct > 90:  # 8th Sept, 2025, A. Mitra - Ultra-critical memory pressure
                adaptive_batch_size = 1
                self.micro_batch_size = 1
                logging.warning(f"Ultra-critical memory pressure ({memory_usage_pct:.1f}%), using single-sample processing")
            elif memory_usage_pct > 85:  # 8th Sept, 2025, A. Mitra - Critical memory pressure
                adaptive_batch_size = max(1, base_batch_size // 16)  # 8th Sept, 2025, A. Mitra - More aggressive reduction
                self.micro_batch_size = min(2, self.micro_batch_size)
                logging.info(f"Critical memory pressure ({memory_usage_pct:.1f}%), reducing batch size from {base_batch_size} to {adaptive_batch_size}")
            elif memory_usage_pct > 75:  # 8th Sept, 2025, A. Mitra - High memory pressure
                adaptive_batch_size = max(1, base_batch_size // 8)  # 8th Sept, 2025, A. Mitra - Increased reduction
                self.micro_batch_size = min(2, self.micro_batch_size)
                logging.info(f"High memory pressure ({memory_usage_pct:.1f}%), reducing batch size from {base_batch_size} to {adaptive_batch_size}")
            elif memory_usage_pct > 60:  # 8th Sept, 2025, A. Mitra - Moderate memory pressure
                adaptive_batch_size = max(2, base_batch_size // 4)
                logging.info(f"Moderate memory pressure ({memory_usage_pct:.1f}%), reducing batch size from {base_batch_size} to {adaptive_batch_size}")
            elif available_memory_mb < 2000:  # 8th Sept, 2025, A. Mitra - Low available memory threshold increased
                adaptive_batch_size = max(1, base_batch_size // 8)
                logging.info(f"Low available memory ({available_memory_mb:.1f}MB), reducing batch size from {base_batch_size} to {adaptive_batch_size}")
            else:
                # Calculate optimal batch size based on available memory  # 8th Sept, 2025, A. Mitra - Use memory efficiently when available
                max_samples_in_memory = int((available_memory_mb * 0.2) / mb_per_sample)  # 8th Sept, 2025, A. Mitra - Reduced from 40% to 20%
                adaptive_batch_size = min(base_batch_size, max(1, max_samples_in_memory))
                if adaptive_batch_size != base_batch_size:
                    logging.info(f"Memory-optimized batch size: {adaptive_batch_size} (from {base_batch_size})")
            
            return adaptive_batch_size
            
        except Exception as e:
            logging.warning(f"Could not calculate adaptive batch size: {e}, using original batch size: {base_batch_size}")
            return base_batch_size
    
    def _predict_with_memory_optimization(self, dataset): 

        """
        # Created Sep 2025 by  A. Mitra 
        Choose between streaming and regular prediction based on dataset size and configuration.
        """
        if not self.memory_optimize:
            logging.info("Memory optimization disabled, using original prediction method")
            return self._predict_original(dataset)
        
        # Estimate dataset size efficiently  # 5th Sept, 2025, A. Mitra - Quick size estimation without full iteration
        try:
            dataset_size = tf.data.experimental.cardinality(dataset).numpy()
            if dataset_size == tf.data.experimental.UNKNOWN_CARDINALITY:
                # For unknown cardinality, take a small sample to estimate  # 5th Sept, 2025, A. Mitra - Sampling approach for size estimation
                sample_size = min(1000, self.streaming_threshold // 10)
                sample_count = dataset.take(sample_size).reduce(0, lambda x, _: x + 1).numpy()
                if sample_count == sample_size:
                    logging.info(f"Dataset size > {sample_size}, estimating as large dataset")
                    dataset_size = self.streaming_threshold + 1  # 5th Sept, 2025, A. Mitra - Force streaming for large datasets
                else:
                    dataset_size = sample_count  # 5th Sept, 2025, A. Mitra - Small dataset, exact count
        except Exception as e:
            logging.warning(f"Could not estimate dataset size: {e}, using streaming prediction")
            dataset_size = self.streaming_threshold + 1  # 8th Sept, 2025, A. Mitra - Default to streaming on error
        
        # Intelligent streaming decision based on available memory and dataset characteristics  # 8th Sept, 2025, A. Mitra - Enhanced decision logic
        actual_threshold = self._calculate_intelligent_threshold(dataset_size)
        
        # Choose prediction method based on size and configuration  
        if self.force_streaming or dataset_size > actual_threshold:
            logging.info(f"Using ultra-low memory prediction for dataset size: " \
                         f"{dataset_size} (threshold: {actual_threshold})")
            return self._predict_ultra_low_memory(dataset) 
        else:
            logging.info(f"Using optimized standard prediction for smaller dataset size: {dataset_size}")
            return self._predict_optimized(dataset) 

    def _predict_streaming(self, dataset):  # 5th Sept, 2025, A. Mitra - Memory-efficient streaming prediction
        """
        Streaming prediction that processes dataset in chunks without loading everything into memory.
        Significantly reduces memory usage for large datasets.
        """
        logging.info(f"Using streaming prediction with batch_size={self.batch_size}")
        self.log_memory_usage("Starting streaming prediction")  # 5th Sept, 2025, A. Mitra - Monitor memory at start
        
        # Initialize collections for results  # 5th Sept, 2025, A. Mitra - Collect results incrementally
        all_predictions = []
        all_true_labels = []
        all_snids = []
        
        total_correct = 0
        total_samples = 0
        batch_count = 0
        
        # Create batched datasets for parallel processing  # 5th Sept, 2025, A. Mitra - Separate image/label and metadata processing
        image_label_batches = dataset.map(lambda image, label, *_: (image, label)).batch(self.batch_size)
        metadata_batches = dataset.map(lambda _, label, id_: (label["label"], id_["id"])).batch(self.batch_size)
        
        # Process in streaming chunks  # 5th Sept, 2025, A. Mitra - Process one batch at a time to minimize memory usage
        logging.info("Processing predictions in streaming batches...")
        
        # Zip the batches together for synchronized processing  # 5th Sept, 2025, A. Mitra - Process data and metadata together efficiently
        for (image_batch, _), (true_labels_batch, ids_batch) in zip(image_label_batches, metadata_batches):
            batch_count += 1
            
            # Extract image tensor from the batch structure  # 5th Sept, 2025, A. Mitra - Handle dictionary structure properly
            if isinstance(image_batch, dict):
                images = image_batch['image']  # 5th Sept, 2025, A. Mitra - Extract image tensor from dictionary
            else:
                images = image_batch  # 5th Sept, 2025, A. Mitra - Use directly if already a tensor
            
            batch_size_actual = tf.shape(images)[0].numpy()
            
            # Run prediction on this batch only  # 5th Sept, 2025, A. Mitra - Predict on small batch, not entire dataset
            batch_predictions = self.trained_model.predict(images, verbose=0)  # 5th Sept, 2025, A. Mitra - Direct prediction on image batch
            
            if self.categorical:
                batch_predictions = np.argmax(batch_predictions, axis=1)  # 5th Sept, 2025, A. Mitra - Handle categorical predictions
            batch_predictions = batch_predictions.flatten()
            
            # Extract batch metadata  # 5th Sept, 2025, A. Mitra - Convert TF tensors to numpy arrays
            batch_true_labels = true_labels_batch.numpy()
            batch_snids = ids_batch.numpy()
            
            # Accumulate results  # 5th Sept, 2025, A. Mitra - Collect batch results
            all_predictions.extend(batch_predictions)
            all_true_labels.extend(batch_true_labels)
            all_snids.extend(batch_snids)
            
            # Calculate accuracy incrementally  # 5th Sept, 2025, A. Mitra - Track accuracy without storing all predictions
            batch_correct = np.sum(np.round(batch_predictions) == batch_true_labels)
            total_correct += batch_correct
            total_samples += len(batch_predictions)
            
            # Progress reporting  # 5th Sept, 2025, A. Mitra - Show progress during streaming prediction
            if batch_count % 10 == 0 or self.verbose_data_loading:
                current_acc = total_correct / total_samples if total_samples > 0 else 0
                self.log_memory_usage(f"Processed batch {batch_count}, samples: {total_samples}, accuracy: {current_acc:.3f}")
            
            # Force garbage collection periodically to free memory  # 5th Sept, 2025, A. Mitra - Aggressive memory management
            if batch_count % self.gc_frequency == 0:
                import gc
                gc.collect()  # 5th Sept, 2025, A. Mitra - Free unused memory periodically
                
            # Clear batch variables to help with memory management  # 5th Sept, 2025, A. Mitra - Explicit cleanup
            del images, batch_predictions, batch_true_labels, batch_snids
        
        self.log_memory_usage(f"Completed streaming prediction: {total_samples} samples in {batch_count} batches")
        
        # Calculate final accuracy  # 5th Sept, 2025, A. Mitra - Compute overall accuracy
        final_accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        # Create results dictionary  # 5th Sept, 2025, A. Mitra - Package results for output
        df_dict = {
            'pred_labels': all_predictions,
            'true_labels': all_true_labels, 
            'snid': all_snids
        }
        
        logging.info(f"Streaming prediction completed: {total_samples} samples, accuracy: {final_accuracy:.4f}")
        return df_dict, final_accuracy
        
    def _predict_original(self, dataset):  # 5th Sept, 2025, A. Mitra - Original prediction method (memory-heavy but preserved for compatibility)
        """
        Original prediction method that caches entire dataset in memory.
        Use only when memory optimization is disabled.
        """
        logging.warning("Using original prediction method - high memory usage expected")
        
        dataset = dataset.cache()  # 5th Sept, 2025, A. Mitra - This loads entire dataset into memory
        dataset_no_ids = dataset.map(lambda image, label, *_: (image, label)).batch(self.batch_size)

        # Set verbosity based on config
        predict_verbose = 1 if self.verbose_data_loading else 0
        logging.info(f"Starting prediction on batches (batch_size={self.batch_size})...")
        predictions = self.trained_model.predict(dataset_no_ids, verbose=predict_verbose)

        if self.categorical:
            predictions = np.argmax(predictions, axis=1)
        predictions = predictions.flatten()

        true_labels = dataset.map(lambda _, label, *args: label["label"])
        df_dict = {'pred_labels': predictions, 'true_labels': list(true_labels.as_numpy_iterator())}
        ids = dataset.map(lambda _, label, id_: id_["id"])
        df_dict['snid'] = list(ids.as_numpy_iterator())

        prediction_ints = np.round(predictions)
        acc = float(np.count_nonzero((prediction_ints - list(true_labels.as_numpy_iterator())) == 0)) / len(prediction_ints)

        return df_dict, acc
    
    def _predict_optimized(self, dataset):  # 5th Sept, 2025, A. Mitra - Memory-optimized version for smaller datasets
        """
        Memory-optimized prediction for smaller datasets that can fit in memory but with optimizations.
        Uses less aggressive optimization than streaming but still saves memory compared to original.
        """
        logging.info("Using optimized prediction for smaller dataset")
        self.log_memory_usage("Starting optimized prediction")
        
        # Calculate adaptive batch size based on memory pressure  # 8th Sept, 2025, A. Mitra - Dynamic batch sizing
        adaptive_batch_size = self._calculate_adaptive_batch_size(self.batch_size)
        
        # Don't cache the full dataset, but use prefetching for efficiency  # 5th Sept, 2025, A. Mitra - Balance between memory and performance
        dataset_batched = dataset.map(lambda image, label, *_: (image, label)).batch(adaptive_batch_size).prefetch(2)
        
        # Use micro-batching if enabled for additional memory savings  # 8th Sept, 2025, A. Mitra - Further memory optimization
        if self.enable_micro_batching and adaptive_batch_size > self.micro_batch_size:
            return self._predict_with_micro_batching(dataset, adaptive_batch_size)
        
        # Set verbosity based on config
        predict_verbose = 1 if self.verbose_data_loading else 0
        logging.info(f"Starting optimized prediction on batches (batch_size={adaptive_batch_size})...")
        predictions = self.trained_model.predict(dataset_batched, verbose=predict_verbose)

        if self.categorical:
            predictions = np.argmax(predictions, axis=1)
        predictions = predictions.flatten()

        # Process metadata without caching the full dataset  # 5th Sept, 2025, A. Mitra - Extract metadata efficiently
        true_labels = []
        snids = []
        for item in dataset:
            true_labels.append(item[1]["label"].numpy())
            snids.append(item[2]["id"].numpy())
        
        df_dict = {'pred_labels': predictions, 'true_labels': true_labels, 'snid': snids}

        prediction_ints = np.round(predictions)
        acc = float(np.count_nonzero((prediction_ints - true_labels) == 0)) / len(prediction_ints)
        
        self.log_memory_usage("Completed optimized prediction")
        logging.info(f"Optimized prediction completed: {len(predictions)} samples, accuracy: {acc:.4f}")
        return df_dict, acc

    def _predict_with_micro_batching(self, dataset, adaptive_batch_size):  # 8th Sept, 2025, A. Mitra - Ultra-memory-efficient micro-batching
        """
        Process batches in micro-batches to minimize peak memory usage during prediction.
        """
        logging.info(f"Using micro-batching: batch_size={adaptive_batch_size}, micro_batch_size={self.micro_batch_size}")
        self.log_memory_usage("Starting micro-batched prediction")
        
        # Initialize collections for results
        all_predictions = []
        all_true_labels = []
        all_snids = []
        
        total_correct = 0
        total_samples = 0
        batch_count = 0
        
        # Create batched dataset 
        dataset_batched = dataset.map(lambda image, label, *_: (image, label)).batch(adaptive_batch_size)
        metadata_batched = dataset.map(lambda _, label, id_: (label["label"], id_["id"])).batch(adaptive_batch_size)
        
        # Process each batch in micro-batches  # 8th Sept, 2025, A. Mitra - Split large batches into smaller chunks
        for (image_batch, _), (true_labels_batch, ids_batch) in zip(dataset_batched, metadata_batched):
            batch_count += 1
            
            # Extract tensors  # 8th Sept, 2025, A. Mitra - Handle dictionary structure
            if isinstance(image_batch, dict):
                images = image_batch['image']
            else:
                images = image_batch
            
            batch_size_actual = tf.shape(images)[0].numpy()
            
            # Process this batch in micro-batches  # 8th Sept, 2025, A. Mitra - Split batch to reduce peak memory
            batch_predictions = []
            
            for i in range(0, batch_size_actual, self.micro_batch_size):
                end_idx = min(i + self.micro_batch_size, batch_size_actual)
                micro_batch = images[i:end_idx]
                
                # Process micro-batch  # 8th Sept, 2025, A. Mitra - Minimal memory footprint
                micro_predictions = self.trained_model.predict(micro_batch, verbose=0)
                batch_predictions.append(micro_predictions)
                
                # Clear micro-batch immediately  # 8th Sept, 2025, A. Mitra - Aggressive memory management
                del micro_batch, micro_predictions
                
                # Garbage collect frequently during micro-batching  # 8th Sept, 2025, A. Mitra - Keep memory usage minimal
                if i % (self.micro_batch_size * 4) == 0:
                    import gc
                    gc.collect()
            
            # Combine micro-batch predictions  # 8th Sept, 2025, A. Mitra - Reconstruct full batch predictions
            batch_predictions = np.concatenate(batch_predictions, axis=0)
            
            if self.categorical:
                batch_predictions = np.argmax(batch_predictions, axis=1)
            batch_predictions = batch_predictions.flatten()
            
            # Extract metadata
            batch_true_labels = true_labels_batch.numpy()
            batch_snids = ids_batch.numpy()
            
            # Accumulate results
            all_predictions.extend(batch_predictions)
            all_true_labels.extend(batch_true_labels)
            all_snids.extend(batch_snids)
            
            # Calculate accuracy
            batch_correct = np.sum(np.round(batch_predictions) == batch_true_labels)
            total_correct += batch_correct
            total_samples += len(batch_predictions)
            
            # Progress reporting and memory monitoring  # 8th Sept, 2025, A. Mitra - Track micro-batching progress
            if batch_count % 5 == 0 or self.verbose_data_loading:
                current_acc = total_correct / total_samples if total_samples > 0 else 0
                self.log_memory_usage(f"Micro-batched {batch_count} batches, samples: {total_samples}, accuracy: {current_acc:.3f}")
            
            # Clear batch variables  # 8th Sept, 2025, A. Mitra - Immediate cleanup
            del images, batch_predictions, batch_true_labels, batch_snids
        
        # Calculate final accuracy
        final_accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        # Create results dictionary
        df_dict = {
            'pred_labels': all_predictions,
            'true_labels': all_true_labels, 
            'snid': all_snids
        }
        
        self.log_memory_usage(f"Completed micro-batched prediction: {total_samples} samples in {batch_count} batches")
        logging.info(f"Micro-batched prediction completed: {total_samples} samples, accuracy: {final_accuracy:.4f}")
        return df_dict, final_accuracy

    def _predict_ultra_low_memory(self, dataset):  # 5th Sept, 2025, A. Mitra - Ultra-low memory approach processing files individually
        """
        Process TFRecord files one by one to minimize memory usage.
        Enhanced with progressive loading and immediate cleanup for maximum memory efficiency.
        """
        logging.info("Using ultra-low memory prediction - processing files individually")
        self.log_memory_usage("Starting ultra-low memory prediction")
        
        # Get the original filenames from the dataset loading
        if type(self.heatmaps_paths) == list:
            filenames = ["{}/{}".format(heatmaps_path, f.name) for heatmaps_path in self.heatmaps_paths for f in os.scandir(heatmaps_path) if "tfrecord" in f.name]
        else:
            filenames = ["{}/{}".format(self.heatmaps_paths, f.name) for f in os.scandir(self.heatmaps_paths) if "tfrecord" in f.name]
        
        # Use ultra-small batch sizes for extreme memory efficiency  # 8th Sept, 2025, A. Mitra - Maximize memory savings
        ultra_batch_size = self._calculate_adaptive_batch_size(self.batch_size)
        if self.ultra_low_memory_mode:
            ultra_batch_size = min(ultra_batch_size, 4)  # 8th Sept, 2025, A. Mitra - Cap batch size in ultra mode
        
        # Initialize collections for results with memory-efficient storage  # 8th Sept, 2025, A. Mitra - Use generators instead of lists
        def result_generator():
            total_correct = 0
            total_samples = 0
            file_count = 0
            
            logging.info(f"Processing {len(filenames)} files with ultra-small batches (size: {ultra_batch_size})")
            
            for filename in filenames:
                file_count += 1
                logging.info(f"Processing file {file_count}/{len(filenames)}: {filename}")
                
                # Process single file at a time  # 5th Sept, 2025, A. Mitra - Load and process one file, then release memory
                try:
                    file_dataset = tf.data.TFRecordDataset([filename], num_parallel_reads=1)
                    file_dataset = file_dataset.map(lambda x: get_images(x, self.input_shape, self.with_z), num_parallel_calls=1)
                    file_dataset = file_dataset.apply(tf.data.experimental.ignore_errors())
                    
                    batch_count_in_file = 0
                    # Process this file in ultra-small batches  # 8th Sept, 2025, A. Mitra - Even smaller batches than before
                    for batch in file_dataset.batch(ultra_batch_size):
                        batch_count_in_file += 1
                        
                        # Immediate processing with minimal memory retention  # 8th Sept, 2025, A. Mitra - Process and release immediately
                        batch_results = self._process_ultra_small_batch(batch, filename)
                        
                        if batch_results is not None:
                            batch_predictions, batch_true_labels, batch_snids = batch_results
                            
                            # Yield results immediately instead of accumulating  # 8th Sept, 2025, A. Mitra - Stream results
                            for pred, true_label, snid in zip(batch_predictions, batch_true_labels, batch_snids):
                                yield pred, true_label, snid
                                
                            # Update counters
                            batch_correct = np.sum(np.round(batch_predictions) == batch_true_labels)
                            total_correct += batch_correct
                            total_samples += len(batch_predictions)
                        
                        # Force immediate cleanup after each batch  # 8th Sept, 2025, A. Mitra - Aggressive memory management
                        del batch
                        if batch_count_in_file % 5 == 0:  # 8th Sept, 2025, A. Mitra - Frequent garbage collection
                            import gc
                            gc.collect()
                    
                    # Clean up file dataset immediately  # 8th Sept, 2025, A. Mitra - Release file data
                    del file_dataset
                    
                    # Progress report per file with memory check  # 8th Sept, 2025, A. Mitra - Monitor memory during processing
                    if file_count % 1 == 0:  # 8th Sept, 2025, A. Mitra - Report after every file
                        current_acc = total_correct / total_samples if total_samples > 0 else 0
                        self.log_memory_usage(f"Completed file {file_count}/{len(filenames)}, samples: {total_samples}, accuracy: {current_acc:.3f}")
                    
                    # Aggressive garbage collection after each file  # 8th Sept, 2025, A. Mitra - Force memory cleanup
                    import gc
                    gc.collect()
                    
                    # Check if we're exceeding memory target  # 8th Sept, 2025, A. Mitra - Dynamic memory monitoring
                    if hasattr(self, 'memory_target_gb'):
                        current_memory_gb = psutil.Process().memory_info().rss / (1024**3)
                        if current_memory_gb > self.memory_target_gb * 1.1:  # 8th Sept, 2025, A. Mitra - 10% tolerance
                            logging.warning(f"Memory usage {current_memory_gb:.1f}GB exceeds target {self.memory_target_gb}GB, forcing additional cleanup")
                            gc.collect()
                            
                except Exception as e:
                    logging.error(f"Error processing file {filename}: {e}")
                    continue
                    
            logging.info(f"Ultra-low memory processing completed: {total_samples} samples, accuracy: {total_correct/total_samples:.4f}")
        
        # Process results progressively to avoid accumulating in memory  # 8th Sept, 2025, A. Mitra - Stream-based results collection
        all_predictions = []
        all_true_labels = []
        all_snids = []
        total_samples = 0
        total_correct = 0
        
        # Process in chunks to balance memory vs. progress tracking  # 8th Sept, 2025, A. Mitra - Chunk-based processing
        chunk_size = 1000  # 8th Sept, 2025, A. Mitra - Process results in chunks
        chunk_predictions = []
        chunk_true_labels = []
        chunk_snids = []
        
        for pred, true_label, snid in result_generator():
            chunk_predictions.append(pred)
            chunk_true_labels.append(true_label)
            chunk_snids.append(snid)
            
            if len(chunk_predictions) >= chunk_size:
                # Process chunk and add to final results  # 8th Sept, 2025, A. Mitra - Batch processing of chunks
                all_predictions.extend(chunk_predictions)
                all_true_labels.extend(chunk_true_labels)
                all_snids.extend(chunk_snids)
                
                chunk_correct = np.sum(np.round(chunk_predictions) == chunk_true_labels)
                total_correct += chunk_correct
                total_samples += len(chunk_predictions)
                
                # Clear chunk data  # 8th Sept, 2025, A. Mitra - Free chunk memory
                chunk_predictions.clear()
                chunk_true_labels.clear() 
                chunk_snids.clear()
                
                # Periodic memory reporting  # 8th Sept, 2025, A. Mitra - Track memory during chunk processing
                if total_samples % (chunk_size * 5) == 0:
                    current_acc = total_correct / total_samples
                    self.log_memory_usage(f"Processed {total_samples} samples, accuracy: {current_acc:.3f}")
        
        # Handle remaining samples in final chunk  # 8th Sept, 2025, A. Mitra - Process final partial chunk
        if chunk_predictions:
            all_predictions.extend(chunk_predictions)
            all_true_labels.extend(chunk_true_labels)
            all_snids.extend(chunk_snids)
            
            chunk_correct = np.sum(np.round(chunk_predictions) == chunk_true_labels)
            total_correct += chunk_correct
            total_samples += len(chunk_predictions)
        
        # Calculate final accuracy
        final_accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        # Create results dictionary
        df_dict = {
            'pred_labels': all_predictions,
            'true_labels': all_true_labels, 
            'snid': all_snids
        }
        
        self.log_memory_usage(f"Ultra-low memory prediction completed: {total_samples} samples")
        logging.info(f"Ultra-low memory prediction completed: {total_samples} samples, accuracy: {final_accuracy:.4f}")
        return df_dict, final_accuracy
    
    def _process_ultra_small_batch(self, batch, filename):  # 8th Sept, 2025, A. Mitra - Process individual batches with extreme memory efficiency
        """
        Process a single ultra-small batch with immediate cleanup.
        """
        try:
            # Extract image data properly  # 5th Sept, 2025, A. Mitra - Handle the data structure
            if len(batch) == 3:  # image, label, id format
                image_data, label_data, id_data = batch
            else:
                logging.warning(f"Unexpected batch structure in file {filename}")
                return None
            
            # Handle dictionary structure for images  # 5th Sept, 2025, A. Mitra - Extract image tensor
            if isinstance(image_data, dict):
                images = image_data['image']
            else:
                images = image_data
            
            batch_size_actual = tf.shape(images)[0].numpy()
            
            # Process with micro-batching if batch is still too large  # 8th Sept, 2025, A. Mitra - Additional micro-batching layer
            if batch_size_actual > self.micro_batch_size:
                predictions_list = []
                for i in range(0, batch_size_actual, self.micro_batch_size):
                    end_idx = min(i + self.micro_batch_size, batch_size_actual)
                    micro_images = images[i:end_idx]
                    
                    # Single micro-batch prediction  # 8th Sept, 2025, A. Mitra - Minimal memory footprint
                    micro_predictions = self.trained_model.predict(micro_images, verbose=0)
                    predictions_list.append(micro_predictions)
                    
                    # Immediate cleanup of micro-batch  # 8th Sept, 2025, A. Mitra - Free memory immediately
                    del micro_images, micro_predictions
                
                # Combine micro-batch results  # 8th Sept, 2025, A. Mitra - Reconstruct batch predictions
                batch_predictions = np.concatenate(predictions_list, axis=0)
                del predictions_list  # 8th Sept, 2025, A. Mitra - Clean up intermediate list
            else:
                # Direct prediction for small batches  # 8th Sept, 2025, A. Mitra - Process small batches directly
                batch_predictions = self.trained_model.predict(images, verbose=0)
            
            if self.categorical:
                batch_predictions = np.argmax(batch_predictions, axis=1)
            batch_predictions = batch_predictions.flatten()
            
            # Extract metadata  # 5th Sept, 2025, A. Mitra - Get labels and IDs
            if isinstance(label_data, dict):
                batch_true_labels = label_data["label"].numpy()
            else:
                batch_true_labels = label_data.numpy()
                
            if isinstance(id_data, dict):
                batch_snids = id_data["id"].numpy()
            else:
                batch_snids = id_data.numpy()
            
            # Return results immediately  # 8th Sept, 2025, A. Mitra - Return for immediate processing
            results = (batch_predictions, batch_true_labels, batch_snids)
            
            # Clear all batch data immediately  # 8th Sept, 2025, A. Mitra - Aggressive cleanup
            del images, image_data, label_data, id_data
            
            return results
            
        except Exception as e:
            logging.error(f"Error processing ultra-small batch: {e}")
            return None

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
    
    def _debug_pause_with_memory_report(self, stage_name):  

        """
        Created Sep 8  2025 by A. Mitra 
        Pause execution and provide detailed memory report for debugging.
        """

        if not self.pause_duration == 0: return # RK fix
            
        # Detailed memory reporting  # 8th Sept, 2025, A. Mitra - Comprehensive memory analysis
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        current_rss_gb = memory_info.rss / (1024**3)
        current_vms_gb = memory_info.vms / (1024**3)
        system_used_gb = system_memory.used / (1024**3)
        system_available_gb = system_memory.available / (1024**3)
        system_total_gb = system_memory.total / (1024**3)
        
        print(f"\n" + "="*80)
        print(f"🔍 MEMORY DEBUG PAUSE: {stage_name}")
        print(f"="*80)
        print(f"Process Memory (RSS):     {current_rss_gb:.2f} GB")
        print(f"Process Memory (VMS):     {current_vms_gb:.2f} GB")
        print(f"System Memory Used:       {system_used_gb:.2f} GB / {system_total_gb:.2f} GB ({system_memory.percent:.1f}%)")
        print(f"System Memory Available:  {system_available_gb:.2f} GB")
        
        # TensorFlow GPU memory if available  # 8th Sept, 2025, A. Mitra - GPU memory info
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                print(f"GPU Devices:              {len(gpus)} GPU(s) detected")
            else:
                print(f"GPU Devices:              No GPUs detected (CPU mode)")
        except:
            print(f"GPU Devices:              Unable to query GPU info")
        
        # Configuration summary  # 8th Sept, 2025, A. Mitra - Show current optimization settings
        print(f"\nOptimization Settings:")
        print(f"  Ultra-low memory mode:  {self.ultra_low_memory_mode}")
        print(f"  Dynamic batch sizing:   {self.enable_dynamic_batch_size}")
        print(f"  Micro-batching:         {self.enable_micro_batching}")
        print(f"  Model quantization:     {self.enable_model_quantization}")
        print(f"  Streaming threshold:    {self.streaming_threshold}")
        print(f"  Current batch size:     {self.batch_size}")
        print(f"  Micro-batch size:       {self.micro_batch_size}")
        
        print(f"\n🕐 Pausing for {self.pause_duration} seconds for memory inspection...")
        print(f"💡 Use 'pmap {process.pid}' or 'htop -p {process.pid}' in another terminal")
        print(f"="*80)
        
        # Pause execution  # 8th Sept, 2025, A. Mitra - Allow time for external memory inspection
        import time
        time.sleep(self.pause_duration)
        
        print(f"🔄 Resuming execution after {stage_name} pause\n")
    
    def _dry_run_memory_baseline(self):  # 8th Sept, 2025, A. Mitra - Measure baseline memory usage without data
        """
        Perform a dry run to measure baseline memory usage without loading data.
        """
        logging.info("🧪 DRY RUN MODE: Testing memory baseline without data loading")
        self.log_memory_usage("Dry run start (before TensorFlow initialization)")
        
        # Pause for initial memory inspection  # 8th Sept, 2025, A. Mitra - Check memory before TF setup
        self._debug_pause_with_memory_report("Initial state (before TensorFlow setup)")
        
        # Initialize TensorFlow strategy (this can use significant memory)  # 8th Sept, 2025, A. Mitra - TF setup memory impact
        logging.info("Initializing TensorFlow distributed strategy...")
        strategy_test = tf.distribute.MirroredStrategy()
        self.log_memory_usage("After TensorFlow strategy initialization")
        self._debug_pause_with_memory_report("TensorFlow strategy initialized")
        
        # Load model if specified (major memory consumer)  # 8th Sept, 2025, A. Mitra - Model loading memory impact
        if self.external_trained_model:
            logging.info(f"Loading trained model from {self.external_trained_model}")
            test_model = models.load_model(self.external_trained_model, 
                                         custom_objects={"Reshape": self.Reshape})
            self.log_memory_usage("After model loading")
            self._debug_pause_with_memory_report("Model loaded")
            
            # Apply quantization if enabled  # 8th Sept, 2025, A. Mitra - Quantization memory impact
            if self.enable_model_quantization:
                logging.info("Applying model quantization...")
                quantized_model = self._apply_model_quantization(test_model)
                self.log_memory_usage("After model quantization")
                self._debug_pause_with_memory_report("Model quantized")
                del test_model  # 8th Sept, 2025, A. Mitra - Clean up original model
            
        # Test empty dataset creation (TensorFlow overhead)  # 8th Sept, 2025, A. Mitra - TF dataset memory overhead
        logging.info("Testing empty dataset creation...")
        empty_dataset = tf.data.Dataset.from_tensor_slices([])
        empty_dataset = empty_dataset.batch(self.batch_size)
        self.log_memory_usage("After empty dataset creation")
        self._debug_pause_with_memory_report("Empty dataset created")
        
        # Memory configuration summary  # 8th Sept, 2025, A. Mitra - Configuration impact analysis
        logging.info("="*60)
        logging.info("🔍 DRY RUN MEMORY ANALYSIS SUMMARY")
        logging.info("="*60)
        
        current_memory_gb = psutil.Process().memory_info().rss / (1024**3)
        logging.info(f"Final baseline memory usage: {current_memory_gb:.2f} GB")
        logging.info(f"Memory target: {self.memory_target_gb} GB")
        logging.info(f"Available for data processing: {max(0, self.memory_target_gb - current_memory_gb):.2f} GB")
        
        if current_memory_gb > self.memory_target_gb * 0.5:  # 8th Sept, 2025, A. Mitra - Warn if baseline is high
            logging.warning(f"⚠️  Baseline memory ({current_memory_gb:.2f}GB) is high relative to target ({self.memory_target_gb}GB)")
            logging.warning("Consider enabling model quantization or reducing batch sizes")
        
        logging.info("✅ Dry run completed - no data was loaded")
        self._debug_pause_with_memory_report("Dry run completed")
        
        return current_memory_gb

    def _load_dataset(self):
        if type(self.heatmaps_paths) == list:
            filenames = ["{}/{}".format(heatmaps_path, f.name) for heatmaps_path in self.heatmaps_paths for f in os.scandir(heatmaps_path) if "tfrecord" in f.name]
        else:
            filenames = ["{}/{}".format(self.heatmaps_paths, f.name) for f in os.scandir(self.heatmaps_paths) if "tfrecord" in f.name]

        np.random.shuffle(filenames)
        logging.info(f"Found {len(filenames)} heatmap files")
        logging.info(f"First randomly shiffled heatmap file: {filenames[0]}")
        
        # Show first few files for debugging  (A.Mitra)
        if len(filenames) > 3:
            logging.info(f"Loading files including: {filenames[:3]}") 
        
        # Calculate total size of files to be loaded 
        total_size_mb = sum(os.path.getsize(f) for f in filenames) / (1024 * 1024)  
        logging.info(f"Total data size to load: {total_size_mb:.1f} MB") 

        # Implement memory-mapped access for large datasets 
        if total_size_mb > 1000 and self.memory_optimize:  
            logging.info("Large dataset detected, using memory-mapped file access")
            raw_dataset = self._create_memory_mapped_dataset(filenames)
        else:
            raw_dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=80)
            logging.info(f"Dataset created with 80 parallel readers")

        # Apply disk caching if enabled  
        if self.enable_disk_caching:
            raw_dataset = self._apply_disk_caching(raw_dataset)

        return raw_dataset


    def _create_memory_mapped_dataset(self, filenames): 

        """
        Created Sep 2025 by A.Mitra
        Create a TensorFlow dataset using memory-mapped files to reduce memory usage.
        """
        try:
            logging.info("Creating memory-mapped dataset for efficient large file access")
            
            # Use single-threaded reading for memory-mapped access  
            raw_dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=1)
            
            # Apply interleaving for better I/O efficiency 
            def create_single_file_dataset(filename):
                return tf.data.TFRecordDataset([filename], num_parallel_reads=1)
            
            filenames_dataset = tf.data.Dataset.from_tensor_slices(filenames)
            raw_dataset = filenames_dataset.interleave(
                create_single_file_dataset,
                cycle_length=min(4, len(filenames)),  # Interleave 4 files at once
                num_parallel_calls=1,                 #  Single-threaded for memory efficiency
                deterministic=False
            )
            
            logging.info(f"Memory-mapped dataset created with interleaved access to {len(filenames)} files")
            return raw_dataset
            
        except Exception as e:
            logging.warning(f"Memory-mapped dataset creation failed: {e}, falling back to standard loading")
            return tf.data.TFRecordDataset(filenames, num_parallel_reads=10)  # 8th Sept, 2025, A. Mitra - Reduced parallel reads as fallback

    def _apply_disk_caching(self, raw_dataset):  # 8th Sept, 2025, A. Mitra - Disk-based caching for intermediate results
        """
        Apply disk caching to avoid reprocessing data in subsequent runs.
        """
        try:
            import hashlib
            
            # Create cache directory if it doesn't exist  # 8th Sept, 2025, A. Mitra - Setup caching infrastructure
            os.makedirs(self.disk_cache_dir, exist_ok=True)
            
            # Generate cache key based on heatmaps path and config  # 8th Sept, 2025, A. Mitra - Unique cache per dataset/config
            cache_key_data = f"{self.heatmaps_paths}_{self.input_shape}_{self.with_z}"
            cache_key = hashlib.md5(cache_key_data.encode()).hexdigest()[:16]
            cache_file = os.path.join(self.disk_cache_dir, f"scone_cache_{cache_key}")
            
            # Check if cache exists and is newer than source files  # 8th Sept, 2025, A. Mitra - Validate cache freshness
            if os.path.exists(cache_file):
                cache_time = os.path.getmtime(cache_file)
                if type(self.heatmaps_paths) == list:
                    newest_source = max(os.path.getmtime(path) for path in self.heatmaps_paths if os.path.exists(path))
                else:
                    newest_source = os.path.getmtime(self.heatmaps_paths) if os.path.exists(self.heatmaps_paths) else 0
                
                if cache_time > newest_source:
                    logging.info(f"Using cached dataset from {cache_file}")
                    return raw_dataset.cache(cache_file)
            
            logging.info(f"Creating disk cache at {cache_file}")
            cached_dataset = raw_dataset.cache(cache_file)
            
            # Trigger cache creation by taking one element  # 8th Sept, 2025, A. Mitra - Force cache population
            try:
                next(iter(cached_dataset.take(1)))
                logging.info("Disk cache successfully created")
            except Exception as cache_error:
                logging.warning(f"Cache creation failed: {cache_error}, continuing without cache")
                return raw_dataset
            
            return cached_dataset
            
        except Exception as e:
            logging.warning(f"Disk caching failed: {e}, continuing without cache")
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
        
        def process_with_progress(x):  
            # Created Sept, 2025, A. Mitra 
            # Inner function to process data with progress tracking
            self._chunk_counter['count'] += 1  #  Increment record counter
            
            # Only report progress during actual processing, not during setup  
            # Avoid confusing progress reports during TF pipeline setup
            # Skip the first record which is just pipeline verification
            # TF processes first record for pipeline verification
            if self._chunk_counter['count'] > 1:
                # Report progress at intervals  
                if self._chunk_counter['count'] % report_interval == 0:
                    elapsed = time.time() - self._chunk_counter['start_time']  # Calculate processing time
                    rate = self._chunk_counter['count'] / elapsed if elapsed > 0 else 0  #  Calculate process rate
                    memory_mb = self.process.memory_info().rss / 1024 / 1024  # Monitor current memory usage
                    progress_pct = (self._chunk_counter['count'] / dataset_size * 100) if dataset_size > 0 else 0  # Calculate completion percentage
                    
                    logging.info(f"Processing record {self._chunk_counter['count']}/{dataset_size} ({progress_pct:.1f}%) | Rate: {rate:.1f} records/sec | Memory: {memory_mb:.1f} MB")  # Comprehensive progress report
                    
                    # Verbose mode shows estimated time remaining  #  Additional info for verbose users
                    if self.verbose_data_loading:
                        remaining = dataset_size - self._chunk_counter['count']  # Calculate remaining records
                        eta = remaining / rate if rate > 0 else 0  # Estimate completion time
                        logging.info(f"  Estimated time remaining: {eta:.1f}s")  # Show ETA to user
                
                # Also report at 25%, 50%, 75% milestones  
                elif dataset_size > 0:
                    progress_pct = self._chunk_counter['count'] / dataset_size * 100  # Calculate current progress percentage
                    if abs(progress_pct - 25) < 0.5 or abs(progress_pct - 50) < 0.5 or abs(progress_pct - 75) < 0.5:  
                        # Check if at milestone
                        elapsed = time.time() - self._chunk_counter['start_time']  # Calculate elapsed time
                        rate = self._chunk_counter['count'] / elapsed if elapsed > 0 else 0  #  Calculate processing rate
                        memory_mb = self.process.memory_info().rss / 1024 / 1024  # Check memory usage at milestone
                        logging.info(f"Progress: {progress_pct:.0f}% ({self._chunk_counter['count']}/{dataset_size}) | Rate: {rate:.1f} records/sec | Memory: {memory_mb:.1f} MB")  # Report milestone progress
            
            return get_images(x, self.input_shape, self.with_z)  # Process  data using existing get_images function
        
        # Apply processing with progress tracking 
        dataset = raw_dataset.map(
            process_with_progress,   # Use progress-tracking wrapper function
            num_parallel_calls=tf.data.AUTOTUNE  # Let TF optimize parallelism automatically
        ).ignore_errors()  # Skip corrupted records gracefully
        
        # Use prefetching for better performance and memory management  - Overlap I/O with computation
        dataset = dataset.prefetch(tf.data.AUTOTUNE)  #  TF manages prefetch buffer size automatically
        
        # Note: actual counting happens during iteration  # Important note about TF's lazy evaluation
        logging.info(f"Dataset pipeline created with {tf.data.AUTOTUNE} parallel processing")  
        
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

    # Sep 2025: updates and monor refactor from A.Mitra: add args to debug and to monitor memory

    parser = argparse.ArgumentParser( 
        description='SCONE (Supernova Classification with Neural Networks) - Train or predict using heatmap data', 
        formatter_class=argparse.RawDescriptionHelpFormatter,  
        epilog="""  # 3rd Sept, 2025, A. Mitra - Detailed help section with examples
Debug flag values:
  0    Production mode (default) - uses legacy retrieve_data  # 3rd Sept, 2025, A. Mitra - Stable default
  1    Verbose logging
  901  Use refactored retrieve_data  # 3rd Sept, 2025, A. Mitra - Enhanced implementation with monitoring
  902  Use refactored retrieve_data with verbose logging  # 3rd Sept, 2025, A. Mitra - Detailed progress tracking

Memory optimization examples:  # 8th Sept, 2025, A. Mitra - Added streaming examples
  %(prog)s --config_path config.yaml --force_streaming           # Always use streaming (memory-efficient)
  %(prog)s --config_path config.yaml --no_streaming             # Disable streaming (speed-optimized)
  %(prog)s --config_path config.yaml --streaming_threshold 5000 # Custom threshold for auto-streaming

Examples:
  %(prog)s --config_path config.yaml
  %(prog)s --config_path config.yaml --debug_flag 902
  %(prog)s --config_path config.yaml --heatmaps_subdir custom_heatmaps
  %(prog)s --config_path config.yaml --force_streaming --debug_flag 902
        """
    )

    msg = 'Path to YAML configuration file (required)'
    parser.add_argument('--config_path',  type=str,  required=True,  help=msg )

    msg = f'Alternative heatmaps subdirectory name (default: {HEATMAPS_SUBDIR_DEFAULT})'
    parser.add_argument('--heatmaps_subdir', type=str, default=HEATMAPS_SUBDIR_DEFAULT, help=msg)

    msg = 'Debug flag for development/testing (0=production, 1=verbose, >1 -> specific test)'
    parser.add_argument('--debug_flag', type=int, default=None,  metavar='N', help=msg)

    msg = 'Force streaming prediction regardless of dataset size (default=True)'
    parser.add_argument('--force_streaming',  action='store_true',  help=msg)

    msg = 'Disable streaming prediction and use regular method (overrides config)'
    parser.add_argument('--no_streaming',  action='store_true', help=msg)

    msg = 'Dataset size threshold for automatic streaming (default: 10000). Overrides config file.'
    parser.add_argument('--streaming_threshold', type=int, default=None, metavar='N', help=msg)

    msg = 'Run in dry run mode to test baseline memory usage without loading data'
    parser.add_argument('--dry_run', action='store_true', help=msg)

    msg = 'pause (seconds) for memory inspection (e.g. using pmap in separate terminal)'
    parser.add_argument('--debug_pause', type=int, default=0, help = msg)

    args = parser.parse_args()
    return args


def override_scone_config(args, scone_config):

    # created Sep 2025 by A.Mitra and R.Kessler

    # Handle debug_flag: command-line overrides config file  
    if args.debug_flag is not None:
        scone_config['debug_flag'] = args.debug_flag  
        logging.info(f"Debug flag set from command line: {args.debug_flag}") 
    elif 'debug_flag' not in scone_config:
        scone_config['debug_flag'] = 0  # Default value  
    else:
        logging.info(f"Debug flag from config: {scone_config['debug_flag']}")  


    # Handle streaming options: command-line overrides config file  
    if args.force_streaming:
        scone_config['force_streaming'] = True
        logging.info("Force streaming enabled from command line")
    elif args.no_streaming:
        scone_config['force_streaming'] = False
        scone_config['memory_optimize'] = False 
        logging.info("Streaming disabled from command line")
    
    if args.streaming_threshold is not None:
        scone_config['streaming_threshold'] = args.streaming_threshold
        logging.info(f"Streaming threshold set from command line: {args.streaming_threshold}")
    
    # Handle debug and dry run arguments  # 8th Sept, 2025, A. Mitra - Handle new debugging arguments
    if args.dry_run:
        scone_config['dry_run_mode'] = True
        logging.info("Dry run mode enabled from command line")
    
    if args.debug_pause:
        scone_config['pause_duration'] = args.debug_pause
        logging.info(f"Debug pause mode enabled with {args.debug_pause} second duration")
    
    # Validate streaming arguments 
    if args.force_streaming and args.no_streaming:
        logging.error("Error: --force_streaming and --no_streaming cannot be used together")
        sys.exit(1)

    return

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

    # Sep 2025: allow some command line args to override config block
    override_scone_config(args, scone_config)

    # start scone process
    SconeClassifier(scone_config).run()

    # ==== END MAIN ===
