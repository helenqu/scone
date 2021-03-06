.. _quick-start:

******************
Quick Start Guide
******************

.. _environment:

Environment
===============

TODO: make this into a file you can install from
np/scipy/pandas/george/functools/astropy/h5py/tensorflow 2.2+

.. _installation:

Installation
===============

TODO

.. _data_formatting:
Data Formatting
================

Make sure your data is in the right format for heatmap creation: :doc:`data.rst`

.. _config:
Setting Up the Config File
=============================

The configuration for heatmap creation is fully specified by a YAML file. TODO: make example yml file, make quality cuts a part of config

.. _create_heatmaps:
Create Heatmaps
================

First, we remove low quality lightcurves and balance classes using the ``data_cuts.py`` script::
	$ python data_cuts.py --config_path /path/to/your/config.yml
``data_cuts.py`` writes a file of object IDs that passed quality cuts and class-balancing. These are the IDs of the supernovae that we'll be making heatmaps for next.
**Note:** this can be time-intensive for large datasets, so run this in a ``screen`` session or via a workload manager, such as Slurm.

Next, create the heatmaps::
	$ python create_heatmaps_tfrecord.py --config_path /path/to/your/config.yml
Note that if there are multiple metadata/observation data files, an ``index`` argument will have to be specified. Automatically looping over a number of indices can easily be done with a shellscript such as this one for 100 metadata/observation data file pairs::
	for i in {0..99}
	do
		python create_heatmaps.py --config_path /path/to/your/config.yml --index $i
	done
