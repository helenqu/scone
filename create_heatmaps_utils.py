import numpy as np
import os, sys
import pandas as pd
import george
from george import kernels
from scipy.optimize import minimize
from functools import partial
from astropy.table import Table
import tensorflow as tf
import yaml
import argparse
import h5py
import time

# HELPER FUNCTIONS
def get_extinction(ebv, wave):
    avu = 3.1*ebv

    x = 10000.0/wave # inverse wavelength in microns   - creates a numpy array
    xv = 1.82
    y = x - 1.82 #another numpy array

    #Creating empty arrays in which to store the final data
    aval=[]
    bval=[]

    #Now need to loop through each indavidual wavlength value
    for i in range(len(x)):

        if (x[i] >= 0.3 and x[i] < 1.1): # For IR data
            a =  0.574*pow(x[i],1.61)
            b = -0.527*pow(x[i],1.61)
            aval.append(a)
            bval.append(b)

        elif (x[i] >= 1.1 and x[i] < 3.3): #For Optical/NIR data
            a = 1. + 0.17699*y[i] - 0.50447*np.power(y[i],2) - 0.02427*np.power(y[i],3) + 0.72085*np.power(y[i],4) + 0.01979*np.power(y[i],5) - 0.77530*np.power(y[i],6) + 0.32999*np.power(y[i],7)
            b = 1.41338*y[i] + 2.28305*np.power(y[i],2) + 1.07233*np.power(y[i],3) - 5.38434*np.power(y[i],4) - 0.62251*np.power(y[i],5) + 5.30260*np.power(y[i],6) - 2.09002*np.power(y[i],7)
            aval.append(a)
            bval.append(b)

        elif (x[i] >= 3.3 and x[i] < 8.0): # For UV data
            if (x[i] >= 5.9):
                fa = -0.04473*np.power(x[i]-5.9,2) - 0.009779*np.power(x[i]-5.9,3)
                fb =  0.21300*np.power(x[i]-5.9,2) + 0.120700*np.power(x[i]-5.9,3)
            else:
                fa = fb = 0.0

            a =  1.752 - 0.316*x[i] - 0.104/(np.power(x[i]-4.67,2) + 0.341) + fa
            b = -3.090 + 1.825*x[i] + 1.206/(np.power(x[i]-4.62,2) + 0.263) + fb

            aval.append(a)
            bval.append(b)

        elif (x[i] >= 8.0 and x[i] <= 10.0):  # For Far-UV data
            a = -1.073 - 0.628*(x[i]-8.) + 0.137*np.power(x[i]-8.,2) - 0.070*np.power(x[i]-8.,3)
            b = 13.670 + 4.257*(x[i]-8.) - 0.420*np.power(x[i]-8.,2) + 0.374*np.power(x[i]-8.,3)

            aval.append(a)
            bval.append(b)
        else:
            a = b = 0.0

            aval.append(a)
            bval.append(b)

    aval = np.array(aval)
    bval = np.array(bval)

    RV = 3.1
    extinct = avu*(aval + bval/RV)

    return extinct

def build_gp(guess_length_scale, sn_data, bands):

    """This is  all  taken from Avacado -
    see https://github.com/kboone/avocado/blob/master/avocado/astronomical_object.py
    In this a 2D matern kernal is used  to  model the transient. The kernel
    width in the wavelength direction is fixed. We fit for the kernel width
    in the time direction"""

#     sn_data = sn_data.reset_index(drop=True)
    mjdall = sn_data['mjd']
    fluxall = sn_data['flux']
    flux_errall = sn_data['flux_err']

    #Want to compute the scale factor that we will use...
    signal_to_noises = np.abs(fluxall) / np.sqrt(flux_errall ** 2 + (1e-2 * np.max(fluxall)) ** 2)
    scale = np.abs(fluxall[np.argmax(signal_to_noises)])

    kernel = (0.5 * scale) ** 2 * kernels.Matern32Kernel([guess_length_scale ** 2, 6000 ** 2], ndim=2)

    gp = george.GP(kernel)
    guess_parameters = gp.get_parameter_vector()

    x_data = np.vstack([mjdall, bands]).T
    gp.compute(x_data, flux_errall)

    def neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.log_likelihood(fluxall)

    def grad_neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(fluxall)

    bounds = [(0, np.log(1000 ** 2))]
    bounds = [(guess_parameters[0] - 10, guess_parameters[0] + 10)] + bounds + [(None, None)]
    # check if result with/without bounds are the same

    try:
        fit_result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like, bounds=bounds)
        gp.set_parameter_vector(fit_result.x)
        gaussian_process = partial(gp.predict, fluxall)
    except ValueError:
        return None

    return gaussian_process

def get_predictions_heatmap(gp, peak_mjd, mjd_bins, wavelength_bins, milkyway_ebv):
    #This makes a new array of time which has 0.5 day cadence
    times = np.linspace(peak_mjd-50, peak_mjd+130, mjd_bins)

    wavelengths = np.linspace(3000.0, 10100.0, wavelength_bins)
    ext = get_extinction(milkyway_ebv, wavelengths)
    ext = np.tile(np.expand_dims(ext, axis=1), len(times))

    time_wavelength_grid = np.transpose([np.tile(times, len(wavelengths)), np.repeat(wavelengths, len(times))])
    predictions, prediction_vars = gp(time_wavelength_grid, return_var=True)
    ext_corrected_predictions = np.array(predictions).reshape(len(wavelengths), len(times)) + ext
    prediction_uncertainties = np.sqrt(prediction_vars).reshape(len(wavelengths), len(times))

    return ext_corrected_predictions, prediction_uncertainties

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def image_example(image_string, label, id):
    feature = {
    'id': _int64_feature(id),
	'label': _int64_feature(label),
	'image_raw': _bytes_feature(image_string),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


# LOAD DATA
def run(config, index):
    # sys.stdout = open(os.path.join(config["heatmaps_path"], "create_heatmaps_{}.log".format(index)), "w+")

    METADATA_PATH = config["metadata_paths"][index]
    LCDATA_PATH = config["lcdata_paths"][index]
    OUTPUT_PATH = config["heatmaps_path"]
    SN_TYPE_ID_MAP = config["sn_type_id_to_name"]
    WAVELENGTH_BINS = config["num_wavelength_bins"]
    MJD_BINS = config["num_mjd_bins"]
    IDS_PATH = config["ids_path"] if "ids_path" in config else None
    CATEGORICAL = config["categorical"]

    print("writing to {}".format(OUTPUT_PATH), flush=True)

    metadata = pd.read_csv(METADATA_PATH, compression="gzip") if os.path.splitext(METADATA_PATH)[1] == ".gz" else pd.read_csv(METADATA_PATH)
    lcdata = pd.read_csv(LCDATA_PATH, compression="gzip") if os.path.splitext(LCDATA_PATH)[1] == ".gz" else pd.read_csv(LCDATA_PATH)
    lcdata_ids = metadata[metadata.true_target.isin(SN_TYPE_ID_MAP.keys())].object_id
    lcdata = Table.from_pandas(lcdata)
    lcdata.add_index('object_id')
    if IDS_PATH:
        ids_file = h5py.File(IDS_PATH, "r")
        ids = [x.decode('utf-8') for x in ids_file["names"]]
        ids_file.close()
        print("job {}: found ids, expect {} total heatmaps".format(index, len(ids)), flush=True)
    else:
        ids = None
        print("job {}: no ids, expect {} total heatmaps".format(index, len(lcdata_ids)), flush=True)

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    done_by_type = {}
    removed_by_type = {}
    done_ids = []

    type_to_int_label = {}

    with tf.io.TFRecordWriter("{}/heatmaps_{}.tfrecord".format(OUTPUT_PATH, index)) as writer:
        for i, sn_id in enumerate(lcdata_ids):
            if i % 1000 == 0:
                print("job {}: processing {} of {}".format(index, i, len(lcdata_ids)), flush=True)
            sn_id = int(sn_id)
            sn_metadata = metadata[metadata.object_id == sn_id]

            if sn_metadata.empty:
                continue
            sn_name = SN_TYPE_ID_MAP[sn_metadata.true_target.iloc[0]]

            if ids and "{}_{}".format(sn_name, sn_id) not in ids:
                continue

            if sn_id in done_ids:
                done_by_type[sn_name] = 1 if sn_name not in done_by_type else done_by_type[sn_name] + 1
                continue
            sn_data = lcdata.loc['object_id', sn_id]['mjd', 'flux', 'flux_err', 'passband']
            peak_mjd = sn_metadata['true_peakmjd'].iloc[0]
            filter_to_band_number = {
                "b'u '": 0,
                "b'g '": 1,
                "b'r '": 2,
                "b'i '": 3,
                "b'z '": 4,
                "b'Y '": 5
            }
            replaced_passband = [filter_to_band_number[elem] if elem in filter_to_band_number else int(elem) for elem in sn_data['passband']]
            sn_data['passband'] = replaced_passband
            
            sn_data.add_row([min(sn_data['mjd'])-100, 0, 0, 0])
            sn_data.add_row([max(sn_data['mjd'])+100, 0, 0, 0])
            band_to_wave = {
                0: 3670.69,
                1: 4826.85,
                2: 6223.24,
                3: 7545.98,
                4: 8590.90,
                5: 9710.28
            }

            if "z_obs" in sn_metadata.columns: # augmented
                z_obs = sn_metadata['z_obs'].iloc[0]
                z_sim = sn_metadata['z_sim'].iloc[0]
                band_to_wave = {k:v*(1+z_sim/1+z_obs) for k,v in band_to_wave}

            wave = [band_to_wave[elem] for elem in sn_data['passband']]

            gp = build_gp(20, sn_data, wave)
            if gp == None:
                removed_by_type[sn_name] = 1 if sn_name not in removed_by_type else removed_by_type[sn_name] + 1
                continue

            milkyway_ebv = sn_metadata['mwebv'].iloc[0]
            z = sn_metadata['true_z'].iloc[0]

            predictions, prediction_errs = get_predictions_heatmap(gp, peak_mjd, MJD_BINS, WAVELENGTH_BINS, milkyway_ebv)
            heatmap = np.dstack((predictions, prediction_errs))

            if sn_name not in type_to_int_label:
                if sn_name == "SNIa" or sn_name == "Ia":
                    type_to_int_label[sn_name] = 0 if CATEGORICAL else 1
                else:
                    if CATEGORICAL:
                        type_to_int_label[sn_name] = (max(type_to_int_label.values()) if len(type_to_int_label.values()) > 0 else 0) + 1
                    else:
                        sn_name = "non-Ia"
                        type_to_int_label[sn_name] = 0

            writer.write(image_example(heatmap.flatten().tobytes(), type_to_int_label[sn_name], sn_id))
            done_ids.append(sn_id)
            done_by_type[sn_name] = 1 if sn_name not in done_by_type else done_by_type[sn_name] + 1

    with open("{}/done.log".format(OUTPUT_PATH), "a+") as f:
        f.write("####### JOB {} REPORT #######\n".format(index))
        f.write("type name mapping to integer label used for classification: {}".format(type_to_int_label))
        # TODO: fix the output to done.log
        f.write(str(done_by_type).replace("'", "") + "\n")
        total = 0
        for v in done_by_type.values():
            total += v
        f.write("done: {}\n".format(total))
        f.write("removed: {}\n".format(str(removed_by_type).replace("'", "")))

########### START MAIN FUNCTION ########### 
# parser = argparse.ArgumentParser(description='create heatmaps from lightcurve data')
# parser.add_argument('--config_path', type=str, help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
# parser.add_argument('--index', type=int, default=0, help='integer job index / slurm array job id')
# args = parser.parse_args()

# # LOAD CONFIG
# def load_config(config_path):
#     with open(config_path, "r") as cfgfile:
#         config = yaml.load(cfgfile)
#     return config

# config = load_config(args.config_path)
# run(config, args.index)
