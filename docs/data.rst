.. _data_format:

******************
Lightcurve Data Format
******************

Lightcurve data that will be used to create heatmaps for CatchyName must be separated into two types of CSV files: *metadata* and *observation data*.
Multiple metadata and observation data files are acceptable (and preferred for large datasets), but there should be a 1-1 correspondence between metadata and observation data files, i.e. all objects in a particular metadata file should correspond to objects in a single observation file.

.. _metadata:

Metadata Format
===============

Metadata is expected in CSV format with a minimum of the following columns:
* ``object_id``: a unique integer ID for each supernova that will be used to cross-reference with the observation data
* ``true_target``: the true type of the supernova, typically an integer (i.e. SNANA uses 101 for SNIa, 122 for SNII, etc)
* ``true_peakmjd``: the time of peak flux for the supernova
* ``mwebv``: milky way extinction

.. _observation_data:

Observation Data Format
===============

Observation data is expected in CSV format with a minimum of the following columns:
* ``object_id``: a unique integer ID for each supernova that will be used to cross-reference with the observation data
* ``mjd``: the time of the observation, in Modified Julian Days (MJD)
* ``passband``: an integer corresponding to the filter used for the observation (i.e. 0 for *u*, 1 for *g*, etc)
* ``flux``: the observed flux
* ``flux_err``: the error on the observed flux
* ``detected_bool``: whether the observation counts as a "detection" for the supernova
	* NOTE: we used S/N > 5 as the detection threshold for each observation
