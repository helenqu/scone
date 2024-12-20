#

import os, sys, logging
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from functools import partial
import george
from george import kernels
from astropy.table import Table
from astropy.io import fits
import yaml


def get_band_to_wave(survey):

    # Apr 2024 RK - this method is for legacy scone only;
    #   refactored scone reads badn_to_wave from FILTERS dictionary in sim-readme.

    band_to_wave = None

    if survey == 'NGRST' or survey == 'ROMAN' or survey == 'WFIRST':
        band_to_wave = {
            "R": 6296.73,
            "Z": 8744.77,
            "Y": 10653.88,
            "J": 12975.72,
            "H": 15848.21,
            "F": 18475.41,
            "K": 21255.00  # RK 4.2024
        }
    if survey == 'LSST' or survey == 'DES':
        # Jun 2024 RK-comment: these <lam> are for 2017-era LSST bands without atmos-trans.
        #    They are quite off for DES (4828, 6435, 7828, 9181). I am NOT fixing it now
        #    to ensure that LEGACY scone behaves the same as before, and it is fine if
        #    both training and predict mode use the same <lam>.
        #    The refactored SCONE reads <lam> from sim-readme, but not for real data ...
        #    so need a more robust fix for real data.
        band_to_wave = {
            "u": 3670.69,  # 2017 era, LSST-approx without atmos-trans
            "g": 4826.85,
            "r": 6223.24,
            "i": 7545.98,
            "z": 8590.90,
            "Y": 9710.28
        }
    if survey == 'DESxxx':  # maybe enable this later for DES ?? 
        band_to_wave = {
            "g": 4828.0,  # from DES-SN5YR (Jun 2024)
            "r": 6435.0,
            "i": 7828.0,
            "z": 9181.0
        }
    if survey == "SDSS":
        band_to_wave = {
            "u": 3561.79,
            "g": 4718.87,
            "r": 6185.19,
            "i": 7499.7,
            "z": 8961.49
        }
    if "PS1" in survey:
        band_to_wave = {
            "g": 4866.46,
            "r": 6214.62,
            "i": 7544.57,
            "z": 8679.47,
            "y": 9633.28
        }
        
    if band_to_wave is None:
        raise ValueError(f"survey {survey} not registered for LEGACY scone! " \
                         f"contact helenqu@sas.upenn.edu")

    logging.info(f"Return hard-wired {survey} band_to_wave = {band_to_wave}")
    return band_to_wave


def read_fits(fname, sn_type_id_to_name, survey_from_config, drop_separators=False):
    """Load SNANA formatted data and cast it to a PANDAS dataframe

    Args:
        fname (str): path + name to PHOT.FITS file
        drop_separators (Boolean): if -777 are to be dropped

    Returns:
        (astropy.table.Table) dataframe from PHOT.FITS file (with ID)
        (pandas.DataFrame) dataframe from HEAD.FITS file
    """

    # load photometry
    lcdata = Table.read(fname, format='fits')

    if len(lcdata) == 0:
        print(f"{fname} empty!!")
        return lcdata, lcdata
    # failsafe
    if lcdata['MJD'][-1] == -777.0:
        lcdata.remove_row(-1)
    if lcdata['MJD'][0] == -777.0:
        lcdata.remove_row(0)

    # load header
    metadata_hdu = fits.open(fname.replace("PHOT.FITS", "HEAD.FITS"))
    survey = survey_from_config if survey_from_config else metadata_hdu[0].header["SURVEY"]

    header = Table.read(fname.replace("PHOT.FITS", "HEAD.FITS"), format="fits")
    df_header = header.to_pandas()
    df_header["SNID"] = df_header["SNID"].astype(np.int32)

    # add SNID to phot for skimming
    arr_ID = np.zeros(len(lcdata), dtype=np.int32)
    # New light curves are identified by MJD == -777.0
    arr_idx = np.where(lcdata["MJD"] == -777.0)[0]
    arr_idx = np.hstack((np.array([0]), arr_idx, np.array([len(lcdata)])))
    # Fill in arr_ID
    for counter in range(1, len(arr_idx)):
        start, end = arr_idx[counter - 1], arr_idx[counter]
        # index starts at zero
        arr_ID[start:end] = df_header.SNID.iloc[counter - 1]
    lcdata["SNID"] = arr_ID

    if drop_separators:
        lcdata = lcdata[lcdata['MJD'] != -777.000]

    KEY_SIM_GENTYPE = 'SIM_GENTYPE'
    KEY_SNTYPE      = 'SNTYPE'   # always there for real data
    is_sim = KEY_SIM_GENTYPE in df_header.columns

    head_col_list = ["SNID", "PEAKMJD", "REDSHIFT_FINAL", "REDSHIFT_FINAL_ERR", "MWEBV"]
    head_col_dict = {"SNID":"object_id", "PEAKMJD": "true_peakmjd", 
                     "REDSHIFT_FINAL": "true_z", "REDSHIFT_FINAL_ERR": "true_z_err", "MWEBV": "mwebv"}

    if is_sim: 
        # append column with true GENTYPE
        head_col_list.append(KEY_SIM_GENTYPE)
        head_col_dict[KEY_SIM_GENTYPE] = "true_target"
    else:
        # for real data, append anything to avoid future crash
        head_col_list.append(KEY_SNTYPE)
        head_col_dict[KEY_SNTYPE] = "true_target"   

    df_header = df_header[head_col_list]
    df_header = df_header.rename(columns=head_col_dict)
    if is_sim:
        df_header.replace({"true_target": sn_type_id_to_name}, inplace=True)

    # check for filter column name from different versions of SNANA
    band_colname = "FLT" if "FLT" in lcdata.columns else "BAND" 
    lcdata = lcdata[["SNID", "MJD", band_colname, "FLUXCAL", "FLUXCALERR"]]
    rename_columns = {"SNID":"object_id", "MJD": "mjd", band_colname: "passband", 
                      "FLUXCAL": "flux", "FLUXCALERR": "flux_err"}
    for old_colname, new_colname in rename_columns.items():
        lcdata.rename_column(old_colname, new_colname)

    # Nov 22 2024 RK - store last char only for filter since there was a
    # recent sim change to write out entire filter name; 
    # e.g previous 'z' is now written as 'LSST-z'          .xyz
    lcdata['passband']  = [ s.strip()[-1:] for s in lcdata['passband']]  
    #sys.exit(f"\n xxx modified lcdata = \n{lcdata}")

    return df_header, lcdata, survey

def build_gp(guess_length_scale, sn_data, bands):
    """This is  all  taken from Avacado -
    see https://github.com/kboone/avocado/blob/master/avocado/astronomical_object.py
    In this a 2D matern kernal is used  to  model the transient. The kernel
    width in the wavelength direction is fixed. We fit for the kernel width
    in the time direction"""

    mjdall      = sn_data['mjd']
    fluxall     = sn_data['flux']
    flux_errall = sn_data['flux_err']

    #Want to compute the scale factor that we will use...
    signal_to_noises = np.abs(fluxall) / np.sqrt(flux_errall**2 + (0.01 * np.max(fluxall))**2)
    scale = np.abs(fluxall[np.argmax(signal_to_noises)])

    kernel = (0.5 * scale)**2 * kernels.Matern32Kernel([guess_length_scale**2, 6000**2], ndim=2)

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

def image_example(image_string, label, id, z, z_err):
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _float_feature(value):
      """Returns a float_list from a float / double."""
      return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    feature = {
        'id': _int64_feature(id),
        'label': _int64_feature(label),
        'z': _float_feature(z),
	'z_err': _float_feature(z_err),
        'image_raw': _bytes_feature(image_string),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

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
