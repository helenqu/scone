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
streaming_threshold: 75000      # Dataset size threshold for mode switching
gc_frequency: 200               # Garbage collection frequency
enable_micro_batching: False    # For extreme memory optimization only
memory_optimize: False          # Additional memory optimizations
```

### Performance Summary

| Mode | Small Datasets (<75K) | Large Datasets (≥75K) | Memory Usage |
|------|----------------------|----------------------|--------------|
| **Balanced (Default)** | 3-4 min | 15-20 min | Adaptive |
| **Fast** | 3-4 min | 3-4 min | ~25-30GB |
| **Memory-Optimized** | 2-3 hours | 2-3 hours | ~5-8GB |

### Key Features

1. **No performance regression**: Small datasets maintain original speed automatically
2. **Intelligent adaptation**: Large datasets automatically get memory-efficient processing
3. **Override available**: Force any mode regardless of dataset size
4. **Debug flag compatible**: Works with all debug flags (0, 901, 902)
