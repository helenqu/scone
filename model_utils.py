#!/usr/bin/env python
#
# Mar 6 2024 RK
#  +  minor refactor in main to accept optional --heatmaps_subdir argument that
#     is useful for side-by-side testing of scone codes or options. This code
#     should still be compatible with both original and refactored scone codes
#
#
# Feb 2 2026 AM
#  +  Fix TypeError in model training: extract dictionary values before passing
#     to model.fit() to resolve "Expected float32, but got label of type 'str'" error
#  +  Fix model saving: save as model.keras file inside trained_model directory
#     to comply with Keras API requirements for file extensions
#

import os, sys, yaml, h5py, h5py, time, json
import argparse, atexit, psutil, gc

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, utils, optimizers

from   data_utils  import *
from   scone_utils import *   # RK - should merge with data_utils ?
import scone_utils as util

# =====================================================
# =====================================================

class SconeClassifier():
    """
    Main classifier for SCONE (Supernova Classification with Neural Networks).

    Implements a CNN-based approach for classifying supernovae from photometric
    heatmap data. Supports both training and prediction modes with intelligent
    memory optimization for large datasets.
    """

    class Reshape(layers.Layer):
        """
        Custom Keras layer for dimension reordering of heatmap tensors.

        Transforms tensor dimensions from TensorFlow's standard format to a
        specific ordering required by the SCONE CNN architecture.

        Transformation:
            Input:  [batch, height, width, channels]  (TensorFlow standard)
            Output: [batch, channels, width, height]   (SCONE architecture)

        This reordering swaps height/width and moves channels from last to second position,
        which may be optimized for the specific convolutional operations in SCONE's architecture.
        """
        def call(self, inputs):
            """
            Apply the dimension reordering transformation.

            Args:
                inputs: Input tensor of shape [batch, height, width, channels]

            Returns:
                Transposed tensor of shape [batch, channels, width, height]
            """
            return tf.transpose(inputs, perm=[0,3,2,1])

        def get_config(self):
            """
            Get layer configuration for model serialization.

            Required by Keras for saving and loading models containing custom layers.

            Returns:
                dict: Empty config dict (layer has no trainable parameters)
            """
            return {}

    def __init__(self, config):
        self.scone_config = config
        self.seed = config.get("seed", 42)

        # Debug flag system for development/testing - MUST be before other settings
        self.debug_flag = config.get('debug_flag', 0)

        # Get memory optimization settings early to determine if we need TF configuration
        self.force_streaming = config.get('force_streaming', False)
        default_memory_optimize = self.debug_flag != 0 if self.debug_flag is not None else False
        self.memory_optimize = config.get('memory_optimize', default_memory_optimize)

        # Configure TF memory BEFORE creating strategy (if needed)
        if self.debug_flag != 0 or self.memory_optimize or self.force_streaming:
            self._configure_tf_memory()  # Apply memory optimization settings

        # Only initialize psutil when debugging or memory monitoring is needed
        if self.debug_flag and self.debug_flag > 0:
            self.process = psutil.Process()  # For memory monitoring
        else:
            self.process = None  # No monitoring for production mode

        self.output_path    = config['output_path']
        self.heatmaps_paths = config['heatmaps_paths'] if 'heatmaps_paths' in config else config['heatmaps_path'] 
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
        
        # Memory optimization settings (already set above for early TF config)
        # BALANCED OPTIMIZATION: Memory-aware settings with acceptable runtime (3-4x nominal)
        self.streaming_threshold = config.get('streaming_threshold', 75000)  # Balanced threshold
        self.gc_frequency = config.get('gc_frequency', 200)  # Moderate GC frequency
        self.enable_dynamic_batch_size = config.get('enable_dynamic_batch_size', False)  # Keep disabled

        # Track if user explicitly set these values (for auto-configuration)  # 30th Oct, 2025, A. Mitra - Smart defaults
        self._user_set_micro_batching = 'enable_micro_batching' in config  # 30th Oct, 2025, A. Mitra
        self._user_set_micro_batch_size = 'micro_batch_size' in config  # 30th Oct, 2025, A. Mitra
        self._user_set_chunk_size = 'chunk_size' in config  # 30th Oct, 2025, A. Mitra

        self.enable_micro_batching = config.get('enable_micro_batching', False)  # Disabled by default, auto-enabled for large datasets
        self.micro_batch_size = config.get('micro_batch_size', 64)  # Larger micro-batches for balance
        self.enable_balanced_mode = config.get('enable_balanced_mode', True)  # DEFAULT: balanced memory/speed mode
        self.balanced_batch_size = config.get('balanced_batch_size', 128)  # Optimal batch size for balance
        self.chunk_size = config.get('chunk_size', 1000)  # Process data in reasonable chunks
        self.enable_model_quantization = config.get('enable_model_quantization', False)  # 8th Sept, 2025, A. Mitra - Enable model quantization for inference
        self.quantization_method = config.get('quantization_method', 'dynamic')  # 8th Sept, 2025, A. Mitra - dynamic, float16, or int8
        self.enable_disk_caching = config.get('enable_disk_caching', False)  # 8th Sept, 2025, A. Mitra - Enable disk-based caching for intermediate results
        self.disk_cache_dir = config.get('disk_cache_dir', '/tmp/scone_cache')  # 8th Sept, 2025, A. Mitra - Directory for disk cache
        self.ultra_low_memory_mode = config.get('ultra_low_memory_mode', False)  # 8th Sept, 2025, A. Mitra - Enable maximum memory reduction
        self.memory_target_gb = config.get('memory_target_gb', 50)  # 8th Sept, 2025, A. Mitra - Target memory usage in GB
        self.dry_run_mode = config.get('dry_run_mode', False)  # 8th Sept, 2025, A. Mitra - Test mode without data loading
        self.debug_pause_mode = config.get('debug_pause_mode', False)  # 8th Sept, 2025, A. Mitra - Add pauses for memory inspection
        self.pause_duration = config.get('pause_duration', 30)  # 8th Sept, 2025, A. Mitra - Pause duration in seconds
        
        # Setup debug modes
        self._setup_debug_modes()  # Configure debug behavior based on flag value

        return
    
    def _configure_tf_memory(self):  # 5th Sept, 2025, A. Mitra - Configure TensorFlow for optimal memory usage
        """Configure TensorFlow memory settings for large dataset processing."""
        # Enable memory growth for GPUs to avoid pre-allocating all memory  # 5th Sept, 2025, A. Mitra - Prevent GPU memory hogging
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)  # 5th Sept, 2025, A. Mitra - Allow gradual GPU memory allocation
                logging.info(f"Configured memory growth for {len(gpus)} GPU(s)")
            except RuntimeError as e:
                logging.warning(f"Failed to configure GPU memory growth: {e}")
        
        # Set mixed precision for memory efficiency (if supported)  # 5th Sept, 2025, A. Mitra - Use less memory per operation
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logging.info("Enabled mixed precision for memory efficiency")
        except Exception as e:
            logging.info(f"Mixed precision not enabled: {e}")
        
        # Configure dataset options for memory efficiency  # 5th Sept, 2025, A. Mitra - Optimize dataset processing
        tf.config.threading.set_intra_op_parallelism_threads(0)  # 5th Sept, 2025, A. Mitra - Use all available CPU cores
        tf.config.threading.set_inter_op_parallelism_threads(0)  # 5th Sept, 2025, A. Mitra - Parallelize between operations
    
    def _setup_debug_modes(self):  # 3rd Sept, 2025, A. Mitra - New method to centralize debug flag configuration for easy maintenance
        """Setup debug modes based on debug_flag value.

        Debug flag meanings:
        0    = Production mode (default) - uses refactored retrieve_data  # 30th Oct, 2025, A. Mitra - Changed default to use refactored implementation
        -901 = Use legacy retrieve_data  

        Note: Use --verbose or -v command-line flag to enable verbose logging with any implementation  # 30th Oct, 2025, A. Mitra - Separate verbose control
        1000+ = Reserved for future debug modes
        """

        # Define debug mode constants for clarity  # 30th Oct, 2025, A. Mitra - Simplified to only implementation choice
        self.DEBUG_MODES = {
            'PRODUCTION': 0,  # Uses refactored (default)
            'LEGACY_RETRIEVE': -901,  # 30th Oct, 2025, A. Mitra - Uses legacy for fallback/comparison
        }
        # Note: verbose_data_loading is now controlled via --verbose command-line flag  # 30th Oct, 2025, A. Mitra
        
        # Apply debug settings  # 30th Oct, 2025, A. Mitra - Simplified to only log implementation choice
        if self.debug_flag == self.DEBUG_MODES['PRODUCTION']:
            logging.info("Debug Mode: Production mode - using REFACTORED retrieve_data")  # 30th Oct, 2025, A. Mitra - Default now uses refactored for stability
        elif self.debug_flag == self.DEBUG_MODES['LEGACY_RETRIEVE']:
            logging.info("Debug Mode: Using LEGACY retrieve_data")  # 30th Oct, 2025, A. Mitra - Legacy for comparison
        else:
            logging.warning(f"Debug Mode: Unknown debug flag {self.debug_flag}, defaulting to PRODUCTION mode")  # 30th Oct, 2025, A. Mitra

        # Verbose logging status (controlled separately via --verbose flag)  # 30th Oct, 2025, A. Mitra
        if self.verbose_data_loading:
            logging.info(f"Verbose logging: ENABLED")  # 30th Oct, 2025, A. Mitra

        ### self.legacy_predict = debug_flag == self.DEBUG_MODES['LEGACY_RETRIEVE']  # 30th Oct, 2025, A. Mitra - Only one legacy flag now ??

    def _adjust_streaming_threshold_for_dataset(self, avg_mb_per_record, total_size_gb, save_original=False):  # 30th Oct, 2025, A. Mitra - Centralized threshold adjustment logic
        """
        Intelligently adjust streaming threshold based on dataset characteristics.

        Args:
            avg_mb_per_record: Average MB per record
            total_size_gb: Total dataset size in GB
            save_original: If True, saves original threshold before adjusting

        Returns:
            bool: True if threshold was adjusted, False otherwise
        """
        if not hasattr(self, '_threshold_already_adjusted') or not self._threshold_already_adjusted:
            # Rule 1: Very large records (>10 MB each) need aggressive streaming
            if avg_mb_per_record > 10:
                adjusted_threshold = min(self.streaming_threshold, max(5000, int(50000 / avg_mb_per_record)))
                if adjusted_threshold < self.streaming_threshold:
                    logging.info(f"Large records detected ({avg_mb_per_record:.1f} MB/record), adjusting streaming threshold from {self.streaming_threshold} to {adjusted_threshold}")
                    if save_original:
                        self._original_streaming_threshold = self.streaming_threshold
                    self.streaming_threshold = adjusted_threshold
                    self._threshold_already_adjusted = True
                    return True

            # Rule 2: Large total size (>40 GB) needs streaming even with smaller records
            elif total_size_gb > 40:
                # Scale threshold based on size: 40-100GB -> 30K, >100GB -> 20K, >200GB -> 10K
                if total_size_gb > 200:
                    adjusted_threshold = min(self.streaming_threshold, 10000)
                elif total_size_gb > 100:
                    adjusted_threshold = min(self.streaming_threshold, 20000)
                else:  # 40-100 GB
                    adjusted_threshold = min(self.streaming_threshold, 30000)

                if adjusted_threshold < self.streaming_threshold:
                    logging.info(f"Large total dataset ({total_size_gb:.1f} GB), adjusting streaming threshold from {self.streaming_threshold} to {adjusted_threshold}")
                    if save_original:
                        self._original_streaming_threshold = self.streaming_threshold
                    self.streaming_threshold = adjusted_threshold
                    self._threshold_already_adjusted = True
                    return True

        return False

    def _estimate_dataset_size(self, dataset, verbose_logging=False):  # 30th Oct, 2025, A. Mitra - Centralized dataset size estimation
        """
        Estimate dataset size with caching and fallback strategies.
        Used by prediction methods (not data loading).

        Args:
            dataset: TensorFlow dataset
            verbose_logging: If True, provides detailed logging

        Returns:
            int: Estimated dataset size
        """
        # Check if we have cached size first
        if hasattr(self, '_dataset_size') and self._dataset_size is not None:
            if verbose_logging:
                logging.info(f"Using known dataset size: {self._dataset_size}")
            return self._dataset_size

        # Estimate dataset size
        try:
            dataset_size = tf.data.experimental.cardinality(dataset).numpy()
            if dataset_size == tf.data.experimental.UNKNOWN_CARDINALITY:
                # For unknown cardinality, take a small sample to estimate
                sample_size = min(1000, self.streaming_threshold // 10)
                sample_count = dataset.take(sample_size).reduce(0, lambda x, _: x + 1).numpy()

                if sample_count == sample_size:
                    if verbose_logging:
                        logging.info(f"Dataset size > {sample_size}, estimating as large dataset")
                    dataset_size = self.streaming_threshold + 1  # Force streaming for large datasets
                else:
                    dataset_size = sample_count  # Small dataset, exact count

            return dataset_size

        except Exception as e:
            if verbose_logging:
                logging.warning(f"Could not estimate dataset size: {e}, using streaming prediction")
            else:
                # Simple exception handling for non-verbose mode
                pass
            return self.streaming_threshold + 1  # Default to streaming on error

    def _auto_configure_for_large_dataset(self, dataset_size, total_size_gb=None):  # 30th Oct, 2025, A. Mitra - Smart auto-configuration
        """
        Automatically configure memory optimization settings for large datasets.
        Only applies if user hasn't explicitly set these values in config.

        Uses fixed, tested values for simplicity: micro_batch_size=16, chunk_size=400

        Args:
            dataset_size: Number of samples in dataset
            total_size_gb: Total dataset size in GB (optional)
        """
        # Determine if this is a large dataset that needs optimization  # 30th Oct, 2025, A. Mitra
        is_large_dataset = (dataset_size >= self.streaming_threshold or
                           self.force_streaming or
                           (total_size_gb is not None and total_size_gb > 40))

        if not is_large_dataset:
            return  # No auto-configuration needed for small datasets

        # Auto-enable micro-batching if not explicitly set by user  # 30th Oct, 2025, A. Mitra
        if not self._user_set_micro_batching:
            self.enable_micro_batching = True
            logging.info(f"Auto-enabled micro-batching for large dataset (size: {dataset_size})")  # 30th Oct, 2025, A. Mitra

        # Auto-configure micro_batch_size to fixed optimal value if not set by user  # 30th Oct, 2025, A. Mitra
        if not self._user_set_micro_batch_size:
            self.micro_batch_size = 16  # Fixed optimal value for streaming
            logging.info(f"Auto-set micro_batch_size=16 for large dataset")  # 30th Oct, 2025, A. Mitra

        # Auto-configure chunk_size to fixed optimal value if not set by user  # 30th Oct, 2025, A. Mitra
        if not self._user_set_chunk_size:
            self.chunk_size = 400  # Fixed optimal value for streaming
            logging.info(f"Auto-set chunk_size=400 for large dataset")  # 30th Oct, 2025, A. Mitra

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
            s.write(f"CPU_SUM:        {t_hr:.3f}  # hr \n")

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
        logging.info(f"Total process time:  {t_minutes:.2f} min")

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
        """
        Main execution method for SCONE classifier.

        Orchestrates the complete workflow based on mode (train or predict):
        - MODE_TRAIN: Loads data, trains model, evaluates on test set
        - MODE_PREDICT: Loads trained model, runs predictions on dataset

        Handles:
        - Dry run mode for memory baseline testing
        - External model loading
        - Memory monitoring and optimization
        - Debug pause points for analysis
        """
        tf.random.set_seed(self.seed)

        self.t_start = time.time()
        self.trained_model = None
        if self.process:  # Only log memory if debugging/monitoring is enabled
            self.log_memory_usage("Initial startup", False)  # Track memory usage at key stages for debugging

        # Handle dry run mode  # 8th Sept, 2025, A. Mitra - Test baseline memory without data
        if self.dry_run_mode:
            baseline_memory = self._dry_run_memory_baseline()
            logging.info(f"🧪 DRY RUN COMPLETED: Baseline memory usage is {baseline_memory:.2f} GB")
            return  # 8th Sept, 2025, A. Mitra - Exit after dry run

        if self.external_trained_model:
            logging.info(f"loading trained model found at {self.external_trained_model}")
            self.trained_model = self._load_trained_model(self.external_trained_model)
            self._debug_pause_with_memory_report("Model loaded from disk")  # 8th Sept, 2025, A. Mitra - Pause after model loading

        if self.mode == MODE_TRAIN:
            self.train_set, self.val_set, self.test_set = self._split_and_retrieve_data()
            self._debug_pause_with_memory_report("Training data loaded")  # 8th Sept, 2025, A. Mitra - Pause after data loading
            self.trained_model, history = self.train()
            history = history.history    

        elif self.mode == MODE_PREDICT:
            t_predict_start = time.time()  # RK                                 
            if self.process:  # Only log memory if debugging/monitoring is enabled
                self.log_memory_usage("Before loading dataset", False) 
            raw_dataset = self._load_dataset()
            if self.process:  # Only log memory if debugging/monitoring is enabled
                self.log_memory_usage("After loading raw dataset", True) 
            
            debug_flag = self.debug_flag
            # Choose retrieve_data implementation based on debug flag  # 30th Oct, 2025, A. Mitra - Simplified logic
            legacy_predict = debug_flag == self.DEBUG_MODES['LEGACY_RETRIEVE']  # 30th Oct, 2025, A. Mitra - Only one legacy flag now

            if legacy_predict:
                # Flag -901 uses legacy implementation
                logging.info("Using LEGACY retrieve_data implementation")
                dataset, size = self._retrieve_data_legacy(raw_dataset)
            elif debug_flag == self.DEBUG_MODES['PRODUCTION'] or debug_flag is None:
                # Default (0) uses refactored implementation  # 30th Oct, 2025, A. Mitra - Refactored is now default
                dataset, size = self._retrieve_data(raw_dataset)  # refactored Sep 2025, A.Mitra
            else:
                sys.exit(f"n ABORT with undefined debug_flag = {debug_flag}")
            
            if self.process:  # Only log memory if debugging/monitoring is enabled
                self.log_memory_usage("Finished processing dataset setup", True)

            # xxxxxxxxx
            #if self.debug_pause_mode:  # Only pause if explicitly enabled
            #    self._debug_pause_with_memory_report("Dataset processing completed")
            # xxxx

            
            logging.info(f"Running scone prediction on full dataset of {size} examples")
            # Store dataset size (number of events) for use in predict method
            self._dataset_size = size
            predict_dict, acc = self.predict(dataset)
            
            self.print_predict_time(t_predict_start, size)  # RK

            # Note: Due to TensorFlow's lazy evaluation, actual data processing   # 3rd Sept, 2025, A. Mitra - Important note for users about TF behavior
            # happens during model.predict() above, not during dataset creation
            
            if self.process:  # Only log memory if debugging/monitoring is enabled
                self.log_memory_usage("After prediction", False)

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
                                  *args: (image["image"], label["label"])).shuffle(100_000).cache().batch(self.batch_size)
        val_set = val_set.map(lambda image, label,
                              *args: (image["image"], label["label"])).shuffle(10_000).cache().batch(self.batch_size)
        logging.info("starting to train")
        history = model.fit(
            train_set,
            epochs=self.num_epochs,
            validation_data=val_set,
            verbose=1,
            class_weight=class_weights if not self.class_balanced else None)

        outdir_train_model = f"{self.output_path}/trained_model"
        os.makedirs(outdir_train_model, exist_ok=True)
        model_file = f"{outdir_train_model}/model.keras"
        model.save(model_file)

        # Oct 2024 RK - make sure output model has g+rw permissions
        cmd_chmod = f"chmod -R g+rw {outdir_train_model}"
        os.system(cmd_chmod)

        # Jun 2024 RK - write mean filter wavelengths to ensure these values
        #               are the same in predict mode.
        self.write_filter_wavelengths(outdir_train_model)  # Jun 2024, RK

        return model, history

    def print_predict_time(self, t_predict_start, nevt_predict):
        # Created Sep 14 2025 by R.Kessler                                 
        # Inputs:                                       
        #    t_predict_start = start time of predictions
        #    nevt_predict    = number of events for which predictions are made

        t_predict_end = time.time()
        dt = (t_predict_end - t_predict_start)  # seconds
        dt_min = dt/60.
        rate_predict = int(float(nevt_predict)/dt)
        logging.info(f"Predict process time = {dt_min:.2f} minutes (predict rate: {rate_predict}/sec)")
        return

    def log_memory_usage(self, code_location, do_pause):

        # Created Sep 2025 by A. Mitra
        # Utility for real-time memory monitoring throughout processing 
        # Inputs: 
        #   code_location: brief name/description of location in code
        #   do_pause     : bool T -> execute pause to enable further interrogation 

        memory_info = self.process.memory_info()
        system_memory = psutil.virtual_memory()

        process_rss_gb   = memory_info.rss / (1024**3)
        system_used_gb   = system_memory.used / (1024**3)
        system_available_gb = system_memory.available / (1024**3)
        system_total_gb  = system_memory.total / (1024**3)

        logging.info(f"MEMORY_MONITOR for {code_location}:\n\t Process Memory: {process_rss_gb:.2f} GB |" \
                     f" System Used: {system_used_gb:.1f}/{system_total_gb:.1f} GB " \
                     f"({system_memory.percent:.1f}%) | Available: {system_available_gb:.1f} GB")

        if do_pause:  # RK 
            self._debug_pause_with_memory_report(code_location)

        return
        # end log_memory_usage 

        

    def _load_trained_model(self, model_path):
        """
        Load a trained model from either a directory (old format) or file (new format).

        Args:
            model_path: Path to model directory or .keras/.h5 file

        Returns:
            Loaded Keras model
        """
        if os.path.isdir(model_path):
            # Old format: directory containing model files
            model_file = os.path.join(model_path, "model.keras")
            if not os.path.exists(model_file):
                # Fallback to h5 format if keras not found
                model_file = os.path.join(model_path, "model.h5")
            if not os.path.exists(model_file):
                raise ValueError(f"No model.keras or model.h5 found in directory: {model_path}")
            logging.info(f"Loading model from: {model_file}")
            return models.load_model(model_file, custom_objects={"Reshape": self.Reshape})
        else:
            # New format: direct path to model file
            logging.info(f"Loading model from: {model_path}")
            return models.load_model(model_path, custom_objects={"Reshape": self.Reshape})

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
            self.trained_model = self._load_trained_model(self.external_trained_model)

        if not self.trained_model:
            raise RuntimeError('model has not been trained! call `train` on the SconeClassifier instance before predict!')

        # BALANCED MODE: New option for memory optimization with reasonable runtime
        if self.enable_balanced_mode:
            logging.info("Using BALANCED prediction mode")
            return self._predict_balanced(dataset)

        # For production mode OR when micro-batching is disabled, use fast legacy prediction
        if self.debug_flag == 0 or not self.enable_micro_batching:
            # Use the original, simple prediction method - identical to nominal version
            return self._predict_legacy(dataset)

        # Only do memory optimization if explicitly enabled via micro_batching
        # Monitor and adjust memory settings if needed
        self._monitor_and_adjust_memory_settings()

        # Apply model quantization for inference if enabled
        if self.enable_model_quantization and self.mode == MODE_PREDICT:
            self.trained_model = self._apply_model_quantization(self.trained_model)

        return self._predict_with_memory_optimization(dataset)  # Intelligent memory-optimized prediction
    
    def _predict_legacy(self, dataset):
        """Legacy prediction method - identical to nominal version."""
        dataset = dataset.cache()  # otherwise the rest of the dataset operations won't return entries in the same order
        dataset_no_ids = dataset.map(lambda image, label, *_: (image, label)).batch(self.batch_size)

        predictions = self.trained_model.predict(dataset_no_ids, verbose=0)

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

    def _predict_balanced(self, dataset):
        """
        BALANCED prediction method - memory-aware with acceptable runtime (3-4x nominal).
        Processes data in manageable chunks without extreme micro-batching.
        Only applies to large datasets; small datasets use fast legacy method.
        """
        # First check dataset size - use fast method for small datasets
        dataset_size = self._estimate_dataset_size(dataset, verbose_logging=False)  # 30th Oct, 2025, A. Mitra - Use centralized estimation

        # INTELLIGENT FIX: Also apply size-based threshold adjustment here if not already done
        # This handles the case where _retrieve_data didn't adjust (e.g., small records but many of them)
        if hasattr(self, '_total_size_mb'):
            avg_mb_per_record = self._total_size_mb / dataset_size if dataset_size > 0 else 0
            total_size_gb = self._total_size_mb / 1024
            self._adjust_streaming_threshold_for_dataset(avg_mb_per_record, total_size_gb, save_original=False)  # 30th Oct, 2025, A. Mitra - Use centralized adjustment

        # Auto-configure memory settings for large datasets if needed  # 30th Oct, 2025, A. Mitra - Smart defaults
        total_size_gb_for_config = self._total_size_mb / 1024 if hasattr(self, '_total_size_mb') else None
        self._auto_configure_for_large_dataset(dataset_size, total_size_gb_for_config)  # 30th Oct, 2025, A. Mitra

        # For small datasets, use fast legacy method even in balanced mode
        if dataset_size < self.streaming_threshold:
            logging.info(f"Small dataset ({dataset_size} < {self.streaming_threshold}), using fast legacy method")
            return self._predict_legacy(dataset)

        # For large datasets, use balanced chunked processing
        logging.info(f"Large dataset ({dataset_size} >= {self.streaming_threshold}), using balanced chunked processing")
        if self.process:
            self.log_memory_usage("Starting balanced prediction for large dataset", False)

        # Use moderate batch size for good balance
        batch_size = self.balanced_batch_size  # Default 128
        chunk_size = self.chunk_size  # Default 1000 samples per chunk

        # Initialize collections
        all_predictions = []
        all_true_labels = []
        all_snids = []

        total_correct = 0
        total_samples = 0
        chunk_count = 0
        batch_count = 0

        # Process dataset in chunks for memory efficiency
        logging.info(f"Balanced mode: batch_size={batch_size}, chunk_size={chunk_size}")

        # Create iterator for chunked processing
        dataset_iter = iter(dataset)
        processing_complete = False

        while not processing_complete:
            chunk_count += 1
            chunk_data = []
            chunk_labels = []
            chunk_ids = []

            # Collect a chunk of data
            try:
                for _ in range(chunk_size):
                    image, label, id_ = next(dataset_iter)
                    # Handle both dictionary and tensor formats
                    if isinstance(image, dict):
                        image_tensor = image['image']
                    else:
                        image_tensor = image
                    chunk_data.append(image_tensor)
                    chunk_labels.append(label["label"].numpy() if hasattr(label["label"], 'numpy') else label["label"])
                    chunk_ids.append(id_["id"].numpy() if hasattr(id_["id"], 'numpy') else id_["id"])
            except StopIteration:
                processing_complete = True

            if not chunk_data:
                break

            # Process chunk in batches
            chunk_predictions = []
            for i in range(0, len(chunk_data), batch_size):
                batch_count += 1
                batch_end = min(i + batch_size, len(chunk_data))
                batch_images = tf.stack(chunk_data[i:batch_end])

                # Predict on batch
                batch_preds = self.trained_model.predict(batch_images, verbose=0)
                if self.categorical:
                    batch_preds = np.argmax(batch_preds, axis=1)
                chunk_predictions.extend(batch_preds.flatten())

                # Moderate frequency progress reporting
                if batch_count % 50 == 0 and self.process:
                    self.log_memory_usage(f"Processed {batch_count} batches, {total_samples + len(chunk_predictions)} samples", False)

            # Accumulate chunk results
            all_predictions.extend(chunk_predictions)
            all_true_labels.extend(chunk_labels)
            all_snids.extend(chunk_ids)

            # Calculate chunk accuracy
            chunk_correct = np.sum(np.round(chunk_predictions) == chunk_labels)
            total_correct += chunk_correct
            total_samples += len(chunk_predictions)

            # Report chunk progress
            if chunk_count % 5 == 0 or self.verbose_data_loading:
                current_acc = total_correct / total_samples if total_samples > 0 else 0
                logging.info(f"Processed chunk {chunk_count:3d}, " \
                             f"NEVT processed: {total_samples:6d}, accuracy: {current_acc:.4f}")

            # Moderate garbage collection
            if self.gc_frequency > 0 and chunk_count % 10 == 0:
                gc.collect()

            # Clear chunk data
            del chunk_data, chunk_labels, chunk_ids, chunk_predictions

        # Final accuracy
        final_accuracy = total_correct / total_samples if total_samples > 0 else 0

        # Create results
        df_dict = {
            'pred_labels': all_predictions,
            'true_labels': all_true_labels,
            'snid': all_snids
        }

        logging.info(f"Balanced prediction completed: {total_samples} samples in {chunk_count} chunks, accuracy: {final_accuracy:.4f}")
        if self.process:
            self.log_memory_usage("Completed balanced prediction", False )

        return df_dict, final_accuracy

    def _monitor_and_adjust_memory_settings(self):  # 8th Sept, 2025, A. Mitra - Real-time memory monitoring and optimization escalation
        """
        Monitor memory usage and automatically escalate optimization strategies.
        """
        if not self.process:  # Skip if monitoring not initialized
            return False

        try:
            current_memory_gb = self.process.memory_info().rss / (1024**3)
            memory = psutil.virtual_memory()
            memory_usage_pct = (memory.used / memory.total) * 100
            available_gb = memory.available / (1024**3)
            
            # Log current memory state
            logging.info(f"Memory monitoring: Current usage: {current_memory_gb:.1f}GB, System: {memory_usage_pct:.1f}%, Available: {available_gb:.1f}GB")
            
            # Auto-escalate optimizations based on memory pressure  # 8th Sept, 2025, A. Mitra - Dynamic optimization adjustment
            escalated = False
            
            if current_memory_gb > self.memory_target_gb or memory_usage_pct > 85:
                if not self.ultra_low_memory_mode:
                    logging.warning(f"Memory target exceeded ({current_memory_gb:.1f}GB > {self.memory_target_gb}GB), enabling ultra-low memory mode")
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
                
                # Reduce batch sizes aggressively  # 8th Sept, 2025, A. Mitra - Emergency batch size reduction
                if current_memory_gb > self.memory_target_gb * 1.2:  # 8th Sept, 2025, A. Mitra - 20% over target
                    self.batch_size = max(1, self.batch_size // 8)
                    self.micro_batch_size = 1
                    logging.warning(f"Emergency memory mode: reducing batch size to {self.batch_size}, micro-batch to 1")
                    escalated = True
                    
                    # Force immediate garbage collection  # 8th Sept, 2025, A. Mitra - Emergency cleanup
                    gc.collect()
                    
                    # Check if memory reduced after cleanup  # 8th Sept, 2025, A. Mitra - Verify cleanup effectiveness
                    new_memory_gb = psutil.Process().memory_info().rss / (1024**3)
                    memory_freed = current_memory_gb - new_memory_gb
                    logging.info(f"Emergency cleanup freed {memory_freed:.1f}GB, new usage: {new_memory_gb:.1f}GB")
            
            if escalated:
                logging.info("Memory optimization settings auto-escalated due to memory pressure")
                self.log_memory_usage("After automatic optimization escalation", False)
            
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
                    # Over target, but use reasonable minimum for performance
                    adaptive_batch_size = max(8, base_batch_size // 4)  # OPTIMIZED: minimum 8, not 1
                    self.micro_batch_size = max(4, self.micro_batch_size // 2)  # OPTIMIZED: minimum 4
                    logging.warning(f"Memory usage {current_usage_gb:.1f}GB exceeds target {self.memory_target_gb}GB, using batch size: {adaptive_batch_size}")
                else:
                    # Calculate batch size to approach but not exceed target  # 8th Sept, 2025, A. Mitra - Conservative approach
                    remaining_memory_mb = target_memory_mb - current_memory_mb
                    max_batch_samples = max(8, int((remaining_memory_mb * 0.5) / mb_per_sample))  # OPTIMIZED: Use 50% of remaining, min 8
                    adaptive_batch_size = min(base_batch_size, max_batch_samples)
                    logging.info(f"Ultra-low memory mode: target {self.memory_target_gb}GB, current {current_usage_gb:.1f}GB, batch size: {adaptive_batch_size}")
                
                return adaptive_batch_size
            
            # Standard aggressive batch size reduction under memory pressure  # 8th Sept, 2025, A. Mitra - Original logic enhanced
            if memory_usage_pct > 95:  # OPTIMIZED: Only at 95%, not 90%
                adaptive_batch_size = max(4, base_batch_size // 8)  # OPTIMIZED: minimum 4, not 1
                self.micro_batch_size = max(2, self.micro_batch_size // 2)
                logging.warning(f"Ultra-critical memory pressure ({memory_usage_pct:.1f}%), using single-sample processing")
            elif memory_usage_pct > 90:  # OPTIMIZED: 90% threshold
                adaptive_batch_size = max(8, base_batch_size // 4)  # OPTIMIZED: Less aggressive
                self.micro_batch_size = min(2, self.micro_batch_size)
                logging.info(f"Critical memory pressure ({memory_usage_pct:.1f}%), reducing batch size from {base_batch_size} to {adaptive_batch_size}")
            elif memory_usage_pct > 80:  # OPTIMIZED: 80% threshold
                adaptive_batch_size = max(16, base_batch_size // 2)  # OPTIMIZED: Only halve batch size
                self.micro_batch_size = min(4, self.micro_batch_size)
                logging.info(f"High memory pressure ({memory_usage_pct:.1f}%), reducing batch size from {base_batch_size} to {adaptive_batch_size}")
            elif memory_usage_pct > 70:  # OPTIMIZED: 70% threshold
                adaptive_batch_size = max(16, base_batch_size // 2)  # OPTIMIZED: Higher minimum
                logging.info(f"Moderate memory pressure ({memory_usage_pct:.1f}%), reducing batch size from {base_batch_size} to {adaptive_batch_size}")
            elif available_memory_mb < 1000:  # OPTIMIZED: Lower threshold for action
                adaptive_batch_size = max(8, base_batch_size // 4)  # OPTIMIZED: Higher minimum
                logging.info(f"Low available memory ({available_memory_mb:.1f}MB), reducing batch size from {base_batch_size} to {adaptive_batch_size}")
            else:
                # Calculate optimal batch size based on available memory  # 8th Sept, 2025, A. Mitra - Use memory efficiently when available
                max_samples_in_memory = int((available_memory_mb * 0.3) / mb_per_sample)  # OPTIMIZED: Use 30% for better performance
                adaptive_batch_size = min(base_batch_size, max(8, max_samples_in_memory))  # OPTIMIZED: Minimum 8
                if adaptive_batch_size != base_batch_size:
                    logging.info(f"Memory-optimized batch size: {adaptive_batch_size} (from {base_batch_size})")
            
            return adaptive_batch_size
            
        except Exception as e:
            logging.warning(f"Could not calculate adaptive batch size: {e}, using original batch size: {base_batch_size}")
            return base_batch_size
    
    def _predict_with_memory_optimization(self, dataset):  # 5th Sept, 2025, A. Mitra - Choose optimal prediction strategy based on dataset size and config
        """
        Choose between streaming and regular prediction based on dataset size and configuration.
        """
        if not self.memory_optimize:
            logging.info("Memory optimization disabled, using original prediction method")
            return self._predict_original(dataset)

        # Use stored dataset size if available (from retrieve_data)
        dataset_size = self._estimate_dataset_size(dataset, verbose_logging=True)  # 30th Oct, 2025, A. Mitra - Use centralized estimation with verbose logging

        # Auto-configure memory settings for large datasets if needed  # 30th Oct, 2025, A. Mitra - Smart defaults
        total_size_gb_for_config = self._total_size_mb / 1024 if hasattr(self, '_total_size_mb') else None
        self._auto_configure_for_large_dataset(dataset_size, total_size_gb_for_config)  # 30th Oct, 2025, A. Mitra

        # For small datasets, ALWAYS use fast legacy method regardless of memory settings
        if dataset_size < self.streaming_threshold and not self.force_streaming:
            logging.info(f"Small dataset ({dataset_size} < {self.streaming_threshold}), using fast legacy prediction")
            return self._predict_legacy(dataset)

        # Intelligent streaming decision based on available memory and dataset characteristics
        actual_threshold = self._calculate_intelligent_threshold(dataset_size)

        # Choose prediction method based on size and configuration
        if self.force_streaming or dataset_size > actual_threshold:
            logging.info(f"Using ultra-low memory prediction for dataset size: {dataset_size} (threshold: {actual_threshold})")
            return self._predict_ultra_low_memory(dataset)  # Process files individually to minimize memory
        else:
            logging.info(f"Using optimized standard prediction for smaller dataset size: {dataset_size}")
            return self._predict_optimized(dataset)  # 5th Sept, 2025, A. Mitra - Memory-optimized version of original method

    def _predict_streaming(self, dataset):  # 5th Sept, 2025, A. Mitra - Memory-efficient streaming prediction
        """
        Streaming prediction that processes dataset in chunks without loading everything into memory.
        Significantly reduces memory usage for large datasets.
        """
        logging.info(f"Using streaming prediction with batch_size={self.batch_size}")
        self.log_memory_usage("Starting streaming prediction", False)  
        
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
            
            # OPTIMIZED: Less frequent progress reporting
            if batch_count % 50 == 0 or (self.verbose_data_loading and batch_count % 10 == 0):
                current_acc = total_correct / total_samples if total_samples > 0 else 0
                self.log_memory_usage(f"Processed batch {batch_count}, samples: {total_samples}, accuracy: {current_acc:.3f}", False)
            
            # OPTIMIZED: Less frequent garbage collection
            if batch_count % (self.gc_frequency * 2) == 0:  # Every 100 batches instead of 50
                gc.collect()  # 5th Sept, 2025, A. Mitra - Free unused memory periodically
                
            # Clear batch variables to help with memory management  # 5th Sept, 2025, A. Mitra - Explicit cleanup
            del images, batch_predictions, batch_true_labels, batch_snids
        
        self.log_memory_usage(f"Completed streaming prediction: {total_samples} samples in {batch_count} batches", False)
        
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
        self.log_memory_usage("Starting optimized prediction", False)
        
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
        
        self.log_memory_usage("Completed optimized prediction", False)
        logging.info(f"Optimized prediction completed: {len(predictions)} samples, accuracy: {acc:.4f}")
        return df_dict, acc

    def _predict_with_micro_batching(self, dataset, adaptive_batch_size):  # 8th Sept, 2025, A. Mitra - Ultra-memory-efficient micro-batching
        """
        Process batches in micro-batches to minimize peak memory usage during prediction.
        """
        logging.info(f"Using micro-batching: batch_size={adaptive_batch_size}, micro_batch_size={self.micro_batch_size}")
        self.log_memory_usage("Starting micro-batched prediction", False)
        
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
                
                # OPTIMIZED: Less frequent GC to reduce overhead - only when really needed
                if self.gc_frequency > 0 and i % (self.micro_batch_size * 20) == 0:  # Much less frequent
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
            
            # OPTIMIZED: Much less frequent progress reporting to dramatically reduce overhead
            # Only report every 1000 batches, or 100 if verbose mode
            if batch_count % 1000 == 0 or (self.verbose_data_loading and batch_count % 100 == 0):
                current_acc = total_correct / total_samples if total_samples > 0 else 0
                if self.process:  # Only log if monitoring enabled
                    self.log_memory_usage(f"Micro-batched {batch_count} batches, samples: {total_samples}, accuracy: {current_acc:.3f}", False)
            
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
        
        self.log_memory_usage(f"Completed micro-batched prediction: {total_samples} samples in {batch_count} batches", False)
        logging.info(f"Micro-batched prediction completed: {total_samples} samples, accuracy: {final_accuracy:.4f}")
        return df_dict, final_accuracy

    def _predict_ultra_low_memory(self, dataset):  # 5th Sept, 2025, A. Mitra - Ultra-low memory approach processing files individually
        """
        Process TFRecord files one by one to minimize memory usage.
        Enhanced with progressive loading and immediate cleanup for maximum memory efficiency.
        """
        logging.info("Using ultra-low memory prediction - processing files individually")
        self.log_memory_usage("Starting ultra-low memory prediction", False)
        
        # Get the original filenames from the dataset loading
        if type(self.heatmaps_paths) == list:
            filenames = ["{}/{}".format(heatmaps_path, f.name) for heatmaps_path in self.heatmaps_paths for f in os.scandir(heatmaps_path) if "tfrecord" in f.name]
        else:
            filenames = ["{}/{}".format(self.heatmaps_paths, f.name) for f in os.scandir(self.heatmaps_paths) if "tfrecord" in f.name]
        
        # Use ultra-small batch sizes for extreme memory efficiency  # 8th Sept, 2025, A. Mitra - Maximize memory savings
        ultra_batch_size = self._calculate_adaptive_batch_size(self.batch_size)
        if self.ultra_low_memory_mode:
            ultra_batch_size = max(8, min(ultra_batch_size, 32))  # OPTIMIZED: Min 8, cap at 32 for better performance
        
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
                    file_dataset = tf.data.TFRecordDataset([filename], num_parallel_reads=4)  # OPTIMIZED: Parallel reads
                    file_dataset = file_dataset.map(lambda x: get_images(x, self.input_shape, self.with_z), num_parallel_calls=tf.data.AUTOTUNE)  # OPTIMIZED: Auto-tuned parallelism
                    file_dataset = file_dataset.apply(tf.data.experimental.ignore_errors())
                    
                    batch_count_in_file = 0
                    # OPTIMIZED: Process with prefetching for better performance
                    for batch in file_dataset.batch(ultra_batch_size).prefetch(2):
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
                        
                        # OPTIMIZED: Less aggressive cleanup - only when really needed
                        del batch
                        if self.gc_frequency > 0 and batch_count_in_file % 50 == 0:  # Much less frequent GC
                            gc.collect()
                    
                    # Clean up file dataset immediately  # 8th Sept, 2025, A. Mitra - Release file data
                    del file_dataset
                    
                    # OPTIMIZED: Less frequent progress reporting - every 10 files instead of every file
                    if file_count % 10 == 0 or (self.verbose_data_loading and file_count % 5 == 0):
                        current_acc = total_correct / total_samples if total_samples > 0 else 0
                        if self.process:  # Only log if monitoring enabled
                            self.log_memory_usage(f"Completed file {file_count}/{len(filenames)}, samples: {total_samples}, accuracy: {current_acc:.3f}", False)
                    
                    # Aggressive garbage collection after each file  # 8th Sept, 2025, A. Mitra - Force memory cleanup
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
                    self.log_memory_usage(f"Processed {total_samples} samples, accuracy: {current_acc:.3f}", False)
        
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
        logging.info(f"Ultra-low memory prediction completed: {total_samples} samples, accuracy: {final_accuracy:.4f}", False)
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
    
    def _debug_pause_with_memory_report(self, stage_name):  # 8th Sept, 2025, A. Mitra - Pause execution for memory analysis
        """
        Pause execution and provide detailed memory report for debugging.
        """
        if not self.debug_pause_mode:
            return
            
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
        time.sleep(self.pause_duration)
        
        print(f"🔄 Resuming execution after {stage_name} pause\n")
    
    def _dry_run_memory_baseline(self):  # 8th Sept, 2025, A. Mitra - Measure baseline memory usage without data
        """
        Perform a dry run to measure baseline memory usage without loading data.
        """
        logging.info("🧪 DRY RUN MODE: Testing memory baseline without data loading")
        self.log_memory_usage("Dry run start (before TensorFlow initialization)", True)

        
        # Initialize TensorFlow strategy (this can use significant memory)  # 8th Sept, 2025, A. Mitra - TF setup memory impact
        logging.info("Initializing TensorFlow distributed strategy...")
        strategy_test = tf.distribute.MirroredStrategy()
        self.log_memory_usage("After TensorFlow strategy initialization", True)
        
        # Load model if specified (major memory consumer)  # 8th Sept, 2025, A. Mitra - Model loading memory impact
        if self.external_trained_model:
            logging.info(f"Loading trained model from {self.external_trained_model}")
            test_model = self._load_trained_model(self.external_trained_model)
            self.log_memory_usage("After model loading", True)
            
            # Apply quantization if enabled  # 8th Sept, 2025, A. Mitra - Quantization memory impact
            if self.enable_model_quantization:
                logging.info("Applying model quantization...")
                quantized_model = self._apply_model_quantization(test_model)
                self.log_memory_usage("After model quantization", True)
                del test_model  
            
        # Test empty dataset creation (TensorFlow overhead) 
        logging.info("Testing empty dataset creation...")
        empty_dataset = tf.data.Dataset.from_tensor_slices([])
        empty_dataset = empty_dataset.batch(self.batch_size)
        self.log_memory_usage("After empty dataset creation", True)
        
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

        np.random.seed(self.seed)  # Dec 19 2025 RK - ensure reproducible file shuffle
        np.random.shuffle(filenames)

        self._num_files = len(filenames)  # Store for size estimation
        logging.info(f"Found {self._num_files} heatmap files")
        logging.info(f"First random heatmap file: {filenames[0]}")
        
        # Show first few files for debugging  # 3rd Sept, 2025, A. Mitra - Help users verify correct files are being loaded
        if len(filenames) > 3:
            logging.info(f"Loading files including: {filenames[:3]}")  # 3rd Sept, 2025, A. Mitra - Display sample of files being processed
        
        # Calculate total size of files to be loaded  # 3rd Sept, 2025, A. Mitra - Inform users about expected data volume
        total_size_mb = sum(os.path.getsize(f) for f in filenames) / (1024 * 1024)  # 3rd Sept, 2025, A. Mitra - Convert bytes to MB for readability
        logging.info(f"Total data size to load: {total_size_mb:.1f} MB")  # 3rd Sept, 2025, A. Mitra - Show total data size to help users plan resource usage

        # Store total size for intelligent streaming threshold adjustment
        self._total_size_mb = total_size_mb

        # Implement memory-mapped access for large datasets  # 8th Sept, 2025, A. Mitra - Reduce memory pressure from file I/O
        if total_size_mb > 1000 and self.memory_optimize:  # 8th Sept, 2025, A. Mitra - Use memory mapping for datasets > 1GB
            logging.info("Large dataset detected, using memory-mapped file access")
            raw_dataset = self._create_memory_mapped_dataset(filenames)
        else:
            raw_dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=80)
            logging.info(f"Dataset created with 80 parallel readers")

        # Apply disk caching if enabled  # 8th Sept, 2025, A. Mitra - Cache processed data to disk to avoid reprocessing
        if self.enable_disk_caching:
            raw_dataset = self._apply_disk_caching(raw_dataset)

        return raw_dataset

    def _create_memory_mapped_dataset(self, filenames):  # 8th Sept, 2025, A. Mitra - Memory-mapped file access for large datasets
        """
        Create a TensorFlow dataset using memory-mapped files to reduce memory usage.
        """
        try:
            logging.info("Creating memory-mapped dataset for efficient large file access")
            
            # Use single-threaded reading for memory-mapped access  # 8th Sept, 2025, A. Mitra - Avoid memory overhead from parallel readers
            raw_dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=1)
            
            # Apply interleaving for better I/O efficiency  # 8th Sept, 2025, A. Mitra - Process files in round-robin fashion
            def create_single_file_dataset(filename):
                return tf.data.TFRecordDataset([filename], num_parallel_reads=1)
            
            filenames_dataset = tf.data.Dataset.from_tensor_slices(filenames)
            raw_dataset = filenames_dataset.interleave(
                create_single_file_dataset,
                cycle_length=min(4, len(filenames)),  # 8th Sept, 2025, A. Mitra - Interleave 4 files at once
                num_parallel_calls=1,  # 8th Sept, 2025, A. Mitra - Single-threaded for memory efficiency
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
        # OPTIMIZED: Smarter dataset size estimation to avoid hanging
        dataset_size = tf.data.experimental.cardinality(raw_dataset).numpy()
        if dataset_size == tf.data.experimental.UNKNOWN_CARDINALITY:
            # Quick sample to estimate size - DON'T count full dataset for large files
            sample_size = 1000  # Fixed sample size
            logging.info(f"Dataset size unknown, sampling {sample_size} records...")
            sample_count = raw_dataset.take(sample_size).reduce(0, lambda x, _: x + 1).numpy()

            if sample_count < sample_size:
                dataset_size = sample_count  # Small dataset, got exact count
            else:
                # Large dataset - estimate based on file count
                if hasattr(self, '_num_files'):
                    # Estimate ~365 samples per file (typical for LSST data)
                    # 160 files * 365 = ~58,400 samples (typical for your dataset)
                    dataset_size = self._num_files * 365
                    logging.info(f"Large dataset detected, estimated ~{dataset_size} records based on {self._num_files} files")
                else:
                    # Conservative estimate for large dataset
                    dataset_size = 50000  # Assume large
                    logging.info(f"Large dataset detected, assuming {dataset_size} records")

        logging.info(f"Total dataset size: {dataset_size} records")

        # INTELLIGENT FIX: Adjust streaming threshold based on actual file sizes
        # If we have large files (>100GB total) but estimated few records, this indicates
        # bloated records that will cause memory issues with legacy method
        if hasattr(self, '_total_size_mb') and not hasattr(self, '_threshold_already_adjusted'):
            avg_mb_per_record = self._total_size_mb / dataset_size if dataset_size > 0 else 0
            total_size_gb = self._total_size_mb / 1024

            # Strategy: Use total dataset size to determine appropriate threshold
            self._adjust_streaming_threshold_for_dataset(avg_mb_per_record, total_size_gb, save_original=True)  # 30th Oct, 2025, A. Mitra - Use centralized adjustment

        # OPTIMIZED: Better threshold and processing choice
        # Use 30K as threshold for simple processing (balanced approach)
        simple_threshold = min(30000, self.streaming_threshold)

        if dataset_size < simple_threshold:
            logging.info(f"Small/medium dataset ({dataset_size} < {simple_threshold}), using fast processing")
            # Use optimized simple method with better parallelism
            dataset = raw_dataset.map(
                lambda x: get_images(x, self.input_shape, self.with_z),
                num_parallel_calls=tf.data.AUTOTUNE  # OPTIMIZED: Auto-tuned parallelism
            ).apply(tf.data.experimental.ignore_errors()).prefetch(tf.data.AUTOTUNE)  # OPTIMIZED: Add prefetching
            return dataset, dataset_size

        # For large datasets, use moderate optimization (not ultra-aggressive)
        logging.info(f"Large dataset ({dataset_size} >= {simple_threshold}), using moderate memory optimization")

        # OPTIMIZED: Skip detailed progress tracking for better performance
        if self.verbose_data_loading:
            self._chunk_counter = {'count': 0, 'start_time': time.time()}
        else:
            self._chunk_counter = None
        # OPTIMIZED: Less frequent reporting for better performance
        if self.verbose_data_loading:
            report_interval = min(500, max(100, dataset_size // 20)) if dataset_size > 0 else 500
            logging.info(f"Verbose mode: Progress will be reported every {report_interval} records")
        else:
            # Non-verbose: minimal reporting
            report_interval = 10000  # Report very infrequently
        
        def process_with_progress(x):  # Inner function to process data
            if self._chunk_counter:
                self._chunk_counter['count'] += 1  # Only count if tracking enabled
            
            # Only report progress if tracking is enabled
            if self._chunk_counter and self._chunk_counter['count'] > 1:
                # Report progress at intervals  # 3rd Sept, 2025, A. Mitra - Regular progress updates
                if self._chunk_counter['count'] % report_interval == 0:
                    elapsed = time.time() - self._chunk_counter['start_time']  # 3rd Sept, 2025, A. Mitra - Calculate processing time
                    rate = self._chunk_counter['count'] / elapsed if elapsed > 0 else 0  # 3rd Sept, 2025, A. Mitra - Calculate processing rate
                    memory_mb = self.process.memory_info().rss / 1024 / 1024 if self.process else 0  # Monitor memory if available
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
                        memory_mb = self.process.memory_info().rss / 1024 / 1024 if self.process else 0  # Check memory at milestone
                        logging.info(f"Progress: {progress_pct:.0f}% ({self._chunk_counter['count']}/{dataset_size}) | Rate: {rate:.1f} records/sec | Memory: {memory_mb:.1f} MB")  # 3rd Sept, 2025, A. Mitra - Report milestone progress
            
            return get_images(x, self.input_shape, self.with_z)  # 3rd Sept, 2025, A. Mitra - Process the actual data using existing get_images function
        
        # OPTIMIZED: Use simple processing if not verbose
        if self._chunk_counter:
            # With progress tracking
            dataset = raw_dataset.map(
                process_with_progress,
                num_parallel_calls=tf.data.AUTOTUNE
            ) # .ignore_errors() # remove this part that crashes on RCC/Midway ?
        else:
            # Without progress tracking (faster)
            dataset = raw_dataset.map(
                lambda x: get_images(x, self.input_shape, self.with_z),
                num_parallel_calls=tf.data.AUTOTUNE
            )  # .ignore_errors() # remove this part that crashes on RCC/Midway ?  
        
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
        
        # Choose retrieve_data implementation based on debug flag  # 30th Oct, 2025, A. Mitra - Simplified logic
        if hasattr(self, 'DEBUG_MODES') and self.debug_flag == self.DEBUG_MODES['LEGACY_RETRIEVE']:
            logging.info("Using LEGACY retrieve_data implementation for training")  # 30th Oct, 2025, A. Mitra - Legacy for comparison
            dataset, size = self._retrieve_data_legacy(raw_dataset)
        else:
            # Default (0) and other flags use refactored  # 30th Oct, 2025, A. Mitra - Changed default to refactored for stability
            logging.info("Using REFACTORED retrieve_data implementation for training")  # 30th Oct, 2025, A. Mitra - Enhanced implementation is now default
            dataset, size = self._retrieve_data(raw_dataset)
        
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
    """
    Specialized SCONE classifier for Type Ia supernova classification.

    Extends SconeClassifier with specific adaptations for binary Ia vs non-Ia
    classification tasks. May include specialized architectures or preprocessing
    optimized for Type Ia identification.

    Note: Apr 16 2024 - Some components may be obsolete, kept for backward compatibility.
    """

    class Reshape(layers.Layer):
        """
        Custom Keras layer for dimension reordering of heatmap tensors.

        Identical to SconeClassifier.Reshape - transforms tensor dimensions from
        TensorFlow's standard format to SCONE CNN architecture format.

        Transformation:
            Input:  [batch, height, width, channels]  (TensorFlow standard)
            Output: [batch, channels, width, height]   (SCONE architecture)

        Note: Apr 16 2024 - May be obsolete, evaluation pending.
        """
        def call(self, inputs):
            """
            Apply the dimension reordering transformation.

            Args:
                inputs: Input tensor of shape [batch, height, width, channels]

            Returns:
                Transposed tensor of shape [batch, channels, width, height]
            """
            return tf.transpose(inputs, perm=[0,3,2,1])

        def get_config(self):
            """
            Get layer configuration for model serialization.

            Returns:
                dict: Empty config dict (layer has no trainable parameters)
            """
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

def check_heatmaps_are_done(config):

    # Created Jan 14 2026 by R.Kessler
    # Check SCONE_SUMMARY under heatmaps/ subdir, and abort if
    #   SCONE_SUMMARY file does not exist, or
    #   STATUS is not 'DONE'.

    heatmap_summ_file = config['heatmaps_path'] + '/' + SCONE_SUMMARY_FILE
    if not os.path.exists(heatmap_summ_file): 
        sys.exit(f"\n ERROR: cannot find expected summary file for heatmaps in \n\t {heatmap_summ_file}")

    heatmap_summ_info = util.load_config_expandvars(heatmap_summ_file, [] )
    status = heatmap_summ_info['STATUS'] 
    if status != 'DONE' :
        sys.exit(f"\n ERROR: heatmap status = {status} (expect DONE) in \n\t {heatmap_summ_file}")
        
    return  # end check_heatmaps_are done

def get_args():

    parser = argparse.ArgumentParser(  
        description='SCONE (Supernova Classification with Neural Networks) - Train or predict using heatmap data',
        formatter_class=argparse.RawDescriptionHelpFormatter,  
        epilog="""
Ordinary Examples:
  %(prog)s --config_path config.yaml                              # Run with default settings (refactored, quiet)
  %(prog)s --config_path config.yaml --verbose                    # Run with verbose logging

Expert/debug Examples (if you don't understand flags below, don't even attempt it) :
  %(prog)s --config_path config.yaml --debug_flag -901            # Run with legacy implementation
  %(prog)s --config_path config.yaml --debug_flag -901 --verbose  # Run legacy with verbose logging
  %(prog)s --config_path config.yaml --heatmaps_subdir custom_heatmaps
  %(prog)s --config_path config.yaml --force_streaming --verbose
  %(prog)s --config_path config.yaml --force_streaming           # Always use streaming (memory-efficient)
  %(prog)s --config_path config.yaml --no_streaming             # Disable streaming (speed-optimized)
  %(prog)s --config_path config.yaml --streaming_threshold 5000 # Custom threshold for auto-streaming

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
                        help='Debug flag  (0=production [default], -901=legacy retreive data). Overrides config file.') 

    parser.add_argument('--verbose', '-v',   # 30th Oct, 2025, A. Mitra - New verbose flag for detailed logging
                       action='store_true',  # 30th Oct, 2025, A. Mitra - Boolean flag
                       help='Enable verbose logging with detailed progress tracking (works with any debug_flag)')  # 30th Oct, 2025, A. Mitra - Clear description

    parser.add_argument('--force_streaming',   # 8th Sept, 2025, A. Mitra - Memory optimization control
                       action='store_true',  # 8th Sept, 2025, A. Mitra - Boolean flag
                       help='debug only: Force streaming prediction regardless of dataset size (overrides config)') 

    parser.add_argument('--no_streaming',   # 8th Sept, 2025, A. Mitra - Disable streaming option
                       action='store_true',  # 8th Sept, 2025, A. Mitra - Boolean flag  
                       help='Disable streaming prediction and use regular method (overrides config)')  # 8th Sept, 2025, A. Mitra - Clear description

    parser.add_argument('--streaming_threshold',   # 8th Sept, 2025, A. Mitra - Configurable threshold
                       type=int, 
                       default=None,  # 8th Sept, 2025, A. Mitra - None allows config file to take precedence
                       metavar='N',  # 8th Sept, 2025, A. Mitra - Clear placeholder
                       help='Dataset size threshold for automatic streaming (default: 10000). Overrides config file.')  # 8th Sept, 2025, A. Mitra - Helpful description

    parser.add_argument('--dry_run',   # 8th Sept, 2025, A. Mitra - Dry run mode for baseline memory testing
                       action='store_true',  # 8th Sept, 2025, A. Mitra - Boolean flag
                       help='Run in dry run mode to test baseline memory usage without loading data')  # 8th Sept, 2025, A. Mitra - Clear description

    parser.add_argument('--debug_pause',   # 8th Sept, 2025, A. Mitra - Debug pause mode for memory inspection
                       action='store_true',  # 8th Sept, 2025, A. Mitra - Boolean flag  
                       help='Enable debug pause mode with memory inspection pauses')  # 8th Sept, 2025, A. Mitra - Clear description

    parser.add_argument('--pause_duration',   # 8th Sept, 2025, A. Mitra - Configurable pause duration
                       type=int, 
                       default=30,  # 8th Sept, 2025, A. Mitra - Default 30 seconds
                       metavar='SECONDS',  # 8th Sept, 2025, A. Mitra - Clear placeholder
                       help='Duration of debug pauses in seconds (default: 30)')  # 8th Sept, 2025, A. Mitra - Helpful description

    args = parser.parse_args()
    return args


# ===============================================
#   MAIN
# ===============================================
if __name__ == "__main__":

    util.setup_logging()

    util.print_job_command()

    logging.info(f"tensorflow version: {tf.__version__}")

    args = get_args()

    # - - - - 
    key_expandvar_list = [ 'output_path', 'trained_model' ]
    scone_config = util.load_config_expandvars(args.config_path, key_expandvar_list )

    # define full path to heatmaps based on subdir
    scone_config['heatmaps_path'] = os.path.join(scone_config['output_path'],args.heatmaps_subdir)

    # Jan 2026 RK - make sure heatmap generation is done
    check_heatmaps_are_done(scone_config)

    # Handle debug_flag: command-line overrides config file  # 3rd Sept, 2025, A. Mitra - Implement priority system for debug flag
    if args.debug_flag is not None:
        scone_config['debug_flag'] = args.debug_flag  # 3rd Sept, 2025, A. Mitra - Command-line takes highest priority
        logging.info(f"Debug flag set from command line: {args.debug_flag}")  # 3rd Sept, 2025, A. Mitra - Inform user about source
    elif 'debug_flag' not in scone_config:
        scone_config['debug_flag'] = 0  # Default value  # 3rd Sept, 2025, A. Mitra - Production mode as default
    else:
        logging.info(f"Debug flag from config: {scone_config['debug_flag']}")  # 3rd Sept, 2025, A. Mitra - Show config file value being used

    # Handle verbose flag: command-line overrides config file  # 30th Oct, 2025, A. Mitra - Separate verbose control
    if args.verbose:
        scone_config['verbose_data_loading'] = True  # 30th Oct, 2025, A. Mitra - Enable verbose logging
        logging.info("Verbose logging enabled from command line")  # 30th Oct, 2025, A. Mitra - Inform user
    elif 'verbose_data_loading' not in scone_config:
        scone_config['verbose_data_loading'] = False  # 30th Oct, 2025, A. Mitra - Default to quiet mode

    # Handle streaming options: command-line overrides config file  # 8th Sept, 2025, A. Mitra - Implement streaming control overrides
    if args.force_streaming:
        scone_config['force_streaming'] = True
        logging.info("Force streaming enabled from command line")
    elif args.no_streaming:
        scone_config['force_streaming'] = False
        scone_config['memory_optimize'] = False  # 8th Sept, 2025, A. Mitra - Disable all memory optimization
        logging.info("Streaming disabled from command line")
    
    if args.streaming_threshold is not None:
        scone_config['streaming_threshold'] = args.streaming_threshold
        logging.info(f"Streaming threshold set from command line: {args.streaming_threshold}")
    
    # Handle debug and dry run arguments  # 8th Sept, 2025, A. Mitra - Handle new debugging arguments
    if args.dry_run:
        scone_config['dry_run_mode'] = True
        logging.info("Dry run mode enabled from command line")
    
    if args.debug_pause:
        scone_config['debug_pause_mode'] = True
        scone_config['pause_duration'] = args.pause_duration
        logging.info(f"Debug pause mode enabled with {args.pause_duration}s pauses")
    
    # Validate streaming arguments  # 8th Sept, 2025, A. Mitra - Prevent conflicting arguments
    if args.force_streaming and args.no_streaming:
        logging.error("Error: --force_streaming and --no_streaming cannot be used together")
        sys.exit(1)

    SconeClassifier(scone_config).run()

    # ==== END MAIN ===
