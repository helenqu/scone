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

