# SCONE: Supernova Classification with a Convolutional Neural Network
This repository contains the code for SCONE ([original paper](https://arxiv.org/abs/2106.04370), [applied to early-time supernova lightcurves](https://arxiv.org/abs/2111.05539)), a convolutional neural network-based framework for photometric supernova classification.

## Installation
`git clone` this repository!

## Requirements
Tensorflow/Keras, [Astropy](https://docs.astropy.org/en/stable/index.html), [George](https://george.readthedocs.io/en/latest/), Pandas, Numpy, Scipy
`requirements.txt` coming soon!

## Overview
SCONE classifies supernovae (SNe) by type using multi-band photometry data (lightcurves)

## Input Data
SCONE takes in supernova (SN) photometry data in the format output by [SNANA](https://github.com/RickKessler/SNANA) simulations.
Photometry data must be separated into two types of files: *metadata* and *observation data*.

Multiple metadata and observation data files are acceptable (and preferred for large datasets), but there should be a 1-1 correspondence between metadata and observation data files, i.e. the observation data for all objects in a particular metadata file should exist in a single corresponding observation file.

### Filenames
Identifying corresponding metadata and observation data files is done through the naming scheme: metadata and observation data files must have the same filename except metadata filenames must include `HEAD` and observation files must include `PHOT`.

i.e. metadata filename: `SN_01_HEAD.FITS`, corresponding observation data filename: `SN_01_PHOT.FITS`

### Metadata Format

Metadata is expected in FITS format with a minimum of the following columns:
* ``SNID``: int, a unique ID for each SN that will be used to cross-reference with the observation data
* ``SNTYPE``: int, representation of the true type of the SN
* ``PEAKMJD``: float, the time of peak flux for the SN in Modified Julian Days (MJD)
* ``MWEBV``: float, milky way extinction
Optional:
* ``REDSHIFT_FINAL``: float, redshift of the SN
* ``REDSHIFT_FINAL_ERR``: float, redshift error

### Observation Data Format

Observation data is expected in FITS format with a minimum of the following columns:
* ``SNID``: int, a unique integer ID for each SN that will be used to cross-reference with the metadata
* ``MJD``: float, the time of the observation, in Modified Julian Days (MJD)
* ``FLT``: string, filter used for the observation (i.e. 'u', 'g')
* ``FLUXCAL``: float, the observed flux
* ``FLUXCAL_ERR``: float, the error on the observed flux

## Quickstart

### 1. Write a configuration file in YAML format

Required fields:
* Either:
  * `input_path` (parent directory of the `PHOT` and `HEAD` files mentioned above), or 
  * a list of `metadata_paths` + a list of `lcdata_paths` (absolute paths to the `HEAD` and `PHOT` files mentioned above, respectively)
* `heatmaps_path`: desired output directory where the heatmaps will be written to
* `mode`: string, `train` or `predict`
* `num_epochs`: int, number of epochs to train for (400 in all paper results)

Optional fields:
* `sn_type_id_to_name`: mapping between integer SN type ID to string name, i.e. `SNII`, defaults to [SNANA default values](https://github.com/helenqu/scone/blob/7f2d2d2d97c114328f9906d6a59d06c1b7129d7e/create_heatmaps/default_gentype_to_typename.yml)
* `class_balanced`: true/false, whether you want class balancing to be done for your input data, defaults to false
* `categorical`: true/false, whether you are doing categorical (by type) classification, defaults to false (i.e. binary Ia vs. non-Ia classification)
* `max_per_type`: int, maximum number of lightcurves per type to keep when performing class balancing (class balancing will take the number of the least abundant class if `max_per_type` not specified)
* `with_z`: true/false, classification with/without redshift information (note that the redshift information for each lightcurve has to be included when making heatmaps, just this option = true is not enough)
* `trained_model`: path, load in a trained model to do prediction with it (goes with `mode: predict`)

### 2. Run `create_heatmaps/run.py` to make heatmaps from your data
`python {/path/to/scone/}create_heatmaps/run.py --config_path {/path/to/config}`
Simply fill in the path to the config file you wrote in the previous step!
This script reads the config file, performs class balancing if desired, and launches jobs to create heatmaps using `sbatch`.
> Note: So far this only works on NERSC! If a different computing system is desired, contact helenqu@sas.upenn.edu.

### 3. When the heatmaps are successfully made, run `run_model.py` to run the model on your new heatmaps
`python {/path/to/scone}/run_model.py --config_path {/path/to/config}`
> Note: So far this only works on NERSC! If a different computing system is desired, contact helenqu@sas.upenn.edu.

## Use with [Pippin](https://github.com/dessn/Pippin/tree/main)
coming soon!

## Performance Optimization and Memory Management

### Performance Modes

SCONE now includes intelligent performance optimization with three modes:

#### 1. **Balanced Mode (Default)** - Intelligent Auto-Selection
Automatically selects the best method based on dataset size:
- **Small datasets (< 75K samples)**: Uses fast method (3-4 minutes) - same as original performance
- **Large datasets (≥ 75K samples)**: Uses memory-efficient chunked processing (15-20 minutes)
- **Memory usage**: ~10-15GB for large datasets, ~25-30GB for small datasets
- No configuration needed - this is the default behavior

#### 2. **Fast Mode** - Maximum Speed
Forces fast processing for all dataset sizes:
- **Runtime**: 3-4 minutes (same as original SCONE)
- **Memory usage**: ~25-30GB
- **Configuration**:
```yaml
enable_balanced_mode: False  # Disable balanced mode for maximum speed
```

#### 3. **Memory Optimization Mode** - Minimum Memory Usage
For extreme memory constraints:
- **Runtime**: 2-3 hours (slower but very memory efficient)
- **Memory usage**: ~5-8GB
- **Configuration**:
```yaml
enable_balanced_mode: False
enable_micro_batching: True  # Enable micro-batching for minimum memory
memory_optimize: True
```

### Configuration Options

Add these to your YAML config file as needed:

```yaml
# Performance settings (all optional - defaults shown)
enable_balanced_mode: True      # Intelligent mode selection (default)
balanced_batch_size: 128        # Batch size for balanced mode
chunk_size: 10000               # Samples per chunk in balanced mode
streaming_threshold: 75000      # Dataset size threshold for mode switching (default: 75000)
gc_frequency: 200               # Garbage collection frequency
enable_micro_batching: False    # For extreme memory optimization only
memory_optimize: False          # Additional memory optimizations
```

### Adaptive Streaming Threshold

The `streaming_threshold` parameter (default: 75000) automatically adapts based on your dataset characteristics:

**Default Behavior:**
- Starts at **75,000 samples** as the threshold for switching to streaming mode
- Automatically adjusts downward for large or memory-intensive datasets

**Automatic Adjustments:**

| Dataset Characteristic | Adjusted Threshold | Reason |
|------------------------|-------------------|---------|
| Records > 10 MB each | 5,000 - 50,000 | Large individual records need earlier streaming |
| Total size > 200 GB | 10,000 | Massive datasets require aggressive memory management |
| Total size > 100 GB | 20,000 | Very large datasets benefit from early streaming |
| Total size > 40 GB | 30,000 | Large datasets use moderate streaming threshold |
| Normal datasets | 75,000 (default) | Standard threshold for typical use cases |

**Example:**
```yaml
streaming_threshold: 75000  # Will auto-adjust to 10000 if dataset is >200GB
```

This adaptive behavior ensures optimal memory usage without requiring manual tuning for different dataset sizes.

### Auto-Configuration for Large Datasets

**NEW**: SCONE now automatically configures advanced memory optimization settings for large datasets. You don't need to manually set these parameters anymore!

**What gets auto-configured:**

When SCONE detects a large dataset (≥75,000 samples, forced streaming, or >40GB total size), it automatically sets:

- **enable_micro_batching**: `true` (enables memory-efficient processing)
- **micro_batch_size**: `16` (optimal tested value)
- **chunk_size**: `400` (optimal tested value)

These are fixed, tested values that work well for all large datasets

**User overrides always take precedence:**
```yaml
# Explicitly set values are never overridden
enable_micro_batching: false  # Will be respected even for large datasets
micro_batch_size: 16          # Will use 16 regardless of dataset size
chunk_size: 400               # Will use 400 regardless of dataset size
```

**Example behavior:**
```bash
# Small dataset (10K samples)
# → No auto-configuration, uses fast defaults

# Large dataset (200K samples, 150GB)
# → Auto-sets: enable_micro_batching=true, micro_batch_size=16, chunk_size=400
# → User sees logs: "Auto-enabled micro-batching for large dataset"

# Large dataset WITH user config
# → Respects all user settings, only auto-configures unset parameters
```

This means **new users don't need to know about these advanced settings** - the system handles it automatically based on their data!

### Performance Summary

| Mode | Small Datasets (<75K) | Large Datasets (≥75K) | Memory Usage |
|------|----------------------|----------------------|--------------|
| **Balanced (Default)** | 3-4 min | 15-20 min | Adaptive |
| **Fast** | 3-4 min | 3-4 min | ~25-30GB |
| **Memory-Optimized** | 2-3 hours | 2-3 hours | ~5-8GB |

### Real-World Benchmarks (LSST Simulations)

Performance measurements from actual LSST simulation datasets:

#### BIASCOR CC Dataset (Smaller)
| Implementation | Memory Allocation | Runtime | Configuration |
|----------------|-------------------|---------|---------------|
| **Legacy** | Default | **0.11 hr** | Standard configuration |
| **Refactored (Default)** | 10 GB | 0.14 hr | `--mem-per-cpu=10GB --ntasks-per-node=1` |

**Analysis**: Slight runtime increase (~27%) but with significantly reduced memory requirements, making it suitable for shared computing environments.

#### BIASCOR Ia Dataset (Larger)
| Implementation | Memory Allocation | Runtime | Configuration |
|----------------|-------------------|---------|---------------|
| **Legacy** | 200 GB | **0.26 hr** | `--mem-per-cpu=200GB --ntasks-per-node=1` |
| **Refactored (Balanced)** | 50 GB | 0.43 hr | `--mem-per-cpu=50GB --ntasks-per-node=1` |
| **Refactored (Memory-Optimized)** | 10 GB | 0.79 hr | `--mem-per-cpu=10GB --ntasks-per-node=1` |

**Analysis**: Clear memory/speed tradeoff:
- **4x memory reduction** (200GB → 50GB) with only ~65% runtime increase
- **20x memory reduction** (200GB → 10GB) with ~3x runtime increase
- Enables running on systems where 200GB memory is unavailable

**Recommendation**: Use default refactored implementation for best balance. For extremely memory-constrained environments, the 10GB configuration still completes in under 1 hour.

### Key Features

1. **No performance regression**: Small datasets maintain original speed automatically
2. **Intelligent adaptation**: Large datasets automatically get memory-efficient processing
3. **Override available**: Force any mode regardless of dataset size
4. **Debug flag compatible**: Works with all debug flags and verbose logging

## Debug Flags and Verbose Logging

### Implementation Selection

SCONE provides debug flags to select between different data retrieval implementations:

- **Flag 0 (Default)**: Uses the refactored `retrieve_data` implementation with enhanced monitoring and memory optimization
- **Flag -901**: Uses the legacy `retrieve_data` implementation for comparison or fallback purposes

**Command-line usage:**
```bash
# Default: uses refactored implementation
python model_utils.py --config_path config.yaml

# Use legacy implementation
python model_utils.py --config_path config.yaml --debug_flag -901
```

**Config file usage:**
```yaml
debug_flag: 0     # Default - refactored implementation
debug_flag: -901  # Legacy implementation
```

### Verbose Logging

Verbose logging can be enabled independently of the implementation choice using the `--verbose` or `-v` flag:

**Command-line usage:**
```bash
# Refactored with verbose logging
python model_utils.py --config_path config.yaml --verbose

# Legacy with verbose logging
python model_utils.py --config_path config.yaml --debug_flag -901 --verbose

# Short form
python model_utils.py --config_path config.yaml -v
```

**Config file usage:**
```yaml
verbose_data_loading: True  # Enable verbose logging
```

Verbose logging provides:
- Detailed progress tracking during data loading
- More frequent batch/chunk progress reports
- Estimated time remaining for long operations
- Enhanced memory usage reporting

### Examples

```bash
# Production mode - default settings (refactored, quiet)
python model_utils.py --config_path config.yaml

# Development mode - verbose logging for debugging
python model_utils.py --config_path config.yaml --verbose

# Testing - compare legacy vs refactored implementations
python model_utils.py --config_path config.yaml --debug_flag -901 --verbose

# Memory-constrained environment with detailed logging
python model_utils.py --config_path config.yaml --force_streaming --verbose
```
