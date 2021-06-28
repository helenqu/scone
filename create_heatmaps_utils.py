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

class CreateHeatmapsManager():
    def __init__(self, config, index):
        self.index = index

        # file paths 
        self.metadata_path = config["metadata_paths"][index]
        self.lcdata_path = config["lcdata_paths"][index]
        self.ids_path = config["ids_path"] if "ids_path" in config else None
        self.output_path = config["heatmaps_path"]
        self.finished_filenames_path = os.path.join(self.output_path, "finished_filenames.csv")

        # heatmap parameters / metadata
        self.sn_type_id_map = config["sn_type_id_to_name"]
        self.wavelength_bins = config["num_wavelength_bins"]
        self.mjd_bins = config["num_mjd_bins"]
        self.has_peakmjd = config.get("has_peakmjd", True)
        self.early_lightcurves = config.get("early_lightcurves", False)

        # heatmap labeling
        self.categorical = config["categorical"]
        self.type_to_int_label = {0: "non-Ia", 1: "SNIa"} if not self.categorical else {v:i for i,v in enumerate(sorted(self.sn_type_id_map.values()))}
        print(f"type to int label: {self.type_to_int_label}")
        
        # keep track of heatmaps

    def load_data(self):
        print(f'Processing file: {self.lcdata_path}')

        if os.path.exists(self.finished_filenames_path):
            finished_filenames = pd.read_csv(self.finished_filenames_path)
            if os.path.basename(self.metadata_path) in finished_filenames:
                print("file has already been processed, exiting")
                sys.exit(0)


        self.metadata = pd.read_csv(self.metadata_path, compression="gzip") if os.path.splitext(self.metadata_path)[1] == ".gz" else pd.read_csv(self.metadata_path)
        metadata_ids = self.metadata[self.metadata.true_target.isin(self.sn_type_id_map.keys())].object_id

        lcdata = pd.read_csv(self.lcdata_path, compression="gzip") if os.path.splitext(self.lcdata_path)[1] == ".gz" else pd.read_csv(self.lcdata_path)
        self.lcdata = Table.from_pandas(lcdata)
        self.lcdata.add_index('object_id')
        self.lcdata_ids = np.intersect1d(self.lcdata['object_id'], metadata_ids)
        
        if self.ids_path:
            ids_file = h5py.File(IDS_PATH, "r")
            self.ids = [x.decode('utf-8') for x in ids_file["names"]]
            ids_file.close()
            print("job {}: found ids, expect {} total heatmaps".format(self.index, len(self.ids)), flush=True)
        else:
            self.ids = None
            print("job {}: no ids, expect {} total heatmaps".format(self.index, len(self.lcdata_ids)), flush=True)


    def run(self):
        self.load_data()
        self.create_heatmaps_early() if self.early_lightcurves else self.create_heatmaps_full()

    def create_heatmaps_early(self):
        days_after_trigger = [5,15,25,50]
        days_before_trigger = -20
        output_paths = [f"{self.output_path}/{days_before_trigger}x{i}_trigger" for i in days_after_trigger]
        mjd_ranges = [[days_before_trigger, i] for i in days_after_trigger]
        self.create_heatmaps(output_paths, "trigger", mjd_ranges, fit_on_full_lc=False)

    def create_heatmaps_full(self):
        print("create heatmaps FULL run")
        self.create_heatmaps([self.output_path], "peakmjd", [[-30, 150]])

    def create_heatmaps(self, output_paths, mjd_zero, mjd_minmaxes, fit_on_full_lc=True):
        for output_path, mjd_minmax in zip(output_paths, mjd_minmaxes):
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            print("writing to {}".format(output_path), flush=True)

            self.done_by_type = {}
            self.removed_by_type = {}
            self.done_ids = []

            with tf.io.TFRecordWriter("{}/heatmaps_{}.tfrecord".format(output_path, self.index)) as writer:
                for i, sn_id in enumerate(self.lcdata_ids):
                    if i % 1000 == 0:
                        print("job {}: processing {} of {}".format(self.index, i, len(self.lcdata_ids)), flush=True)
                    # sn_id = int(sn_id)
                    sn_metadata, sn_name = self._get_sn_data(sn_id)
                    if self._should_skip(sn_id, sn_name, sn_metadata):
                        self.done_by_type[sn_name] = 1 if sn_name not in self.done_by_type else self.done_by_type[sn_name] + 1
                        continue

                    sn_data = self.lcdata.loc['object_id', sn_id]['mjd', 'flux', 'flux_err', 'passband']

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

                    mjd_range = self._calculate_mjd_range(sn_metadata, sn_data, mjd_minmax, mjd_zero, self.has_peakmjd)
                    if not mjd_range:
                        self.removed_by_type[sn_name] = 1 if sn_name not in self.removed_by_type else self.removed_by_type[sn_name] + 1
                        continue

                    if not fit_on_full_lc:
                        mask = np.logical_and(sn_data['mjd'] >= mjd_range[0], sn_data['mjd'] <= mjd_range[1])
                        if not mask.any(): # if all false
                            print("empty sn data after mjd mask", mjd_range, np.min(sn_data['mjd']), np.max(sn_data['mjd']))
                            self.removed_by_type[sn_name] = 1 if sn_name not in self.removed_by_type else self.removed_by_type[sn_name] + 1
                            continue
                        sn_data = sn_data[mask]

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
                    wave = [band_to_wave[elem] for elem in sn_data['passband']]

                    gp = self._build_gp(20, sn_data, wave)
                    if gp == None:
                        self.removed_by_type[sn_name] = 1 if sn_name not in self.removed_by_type else self.removed_by_type[sn_name] + 1
                        continue

                    milkyway_ebv = sn_metadata['mwebv'].iloc[0]
                    z = sn_metadata['true_z'].iloc[0]

                    predictions, prediction_errs = self._get_predictions_heatmap(gp, mjd_range, milkyway_ebv)
                    heatmap = np.dstack((predictions, prediction_errs))

                    if sn_name not in self.type_to_int_label:
                        if self.categorical:
                            print(f"{sn_name} not in SN_TYPE_ID_MAP?? stopping now")
                            break
                        self.type_to_int_label[sn_name] = 1 if sn_name == "SNIa" or sn_name == "Ia" else 0

                    writer.write(self._image_example(heatmap.flatten().tobytes(), self.type_to_int_label[sn_name], sn_id))
                    self.done_ids.append(sn_id)
                    self.done_by_type[sn_name] = 1 if sn_name not in self.done_by_type else self.done_by_type[sn_name] + 1

            if not os.path.exists(self.finished_filenames_path):
                pd.DataFrame({"filenames": [os.path.basename(self.metadata_path)]}).to_csv(self.finished_filenames_path, index=False)
            else:
                finished_filenames = pd.read_csv(self.finished_filenames_path)
                finished_filenames.append({"filenames": os.path.basename(self.metadata_path)}, ignore_index=True).to_csv(self.finished_filenames_path, index=False)

            with open("{}/done.log".format(output_path), "a+") as f:
                f.write("####### JOB {} REPORT #######\n".format(self.index))
                f.write("type name mapping to integer label used for classification: {}".format(self.type_to_int_label))
                # TODO: fix the output to done.log
                f.write(str(self.done_by_type).replace("'", "") + "\n")
                total = 0
                for v in self.done_by_type.values():
                    total += v
                f.write("done: {}\n".format(total))
                f.write("removed: {}\n".format(str(self.removed_by_type).replace("'", "")))

    def _get_sn_data(self, sn_id):
        sn_metadata = self.metadata[self.metadata.object_id == sn_id]

        if sn_metadata.empty:
            print("metadata empty", type(sn_id))
            return sn_metadata, ""

        sn_name = self.sn_type_id_map[sn_metadata.true_target.iloc[0]]
        return sn_metadata, sn_name

    def _should_skip(self, sn_id, sn_name, sn_metadata):
        if sn_metadata.empty:
            return True

        if self.ids and "{}_{}".format(sn_name, sn_id) not in self.ids:
            return True

        if sn_id in self.done_ids:
            return True
        return False

    # HELPER FUNCTIONS
    @staticmethod
    def _calculate_mjd_range(sn_metadata, sn_data, mjd_minmax, mjd_zero, has_peakmjd):
        mjd_min, mjd_max = mjd_minmax
        if mjd_zero == "peakmjd":
            if not has_peakmjd:
                mjd_range = [np.min(sn_data['mjd']), np.max(sn_data['mjd'])]
                return mjd_range
            peak_mjd = sn_metadata['true_peakmjd'].iloc[0]
            mjd_range = [peak_mjd+mjd_min, peak_mjd+mjd_max]
        elif mjd_zero == "trigger":
            sn_data.sort("mjd")
            snrs_by_mjd = [[mjd, flux/flux_err] for mjd, flux, flux_err in sn_data.iterrows('mjd', 'flux', 'flux_err')]
            # snrs_by_mjd = {sn_data.iloc[idx]['mjd']:sn_data.iloc[idx]['flux']/sn_data.iloc[idx]['flux_err'] for idx in range(len(sn_data))}
            detections = [[mjd,snr] for mjd, snr in snrs_by_mjd if snr > 5]
            if len(detections) < 2:
                return 
            first_detection_mjd = detections[0][0]
            # find first detection that occurred more than 1 day after initial detection
            detections = [detection for detection in detections if detection[0] > 1+first_detection_mjd]
            if len(detections) == 0:
                return
            trigger_mjd = detections[0][0]
            mjd_range = [trigger_mjd+mjd_min, trigger_mjd+mjd_max]

        return mjd_range

    @staticmethod
    def _get_extinction(ebv, wave):
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

    @staticmethod
    def _build_gp(guess_length_scale, sn_data, bands):

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

    def _get_predictions_heatmap(self, gp, mjd_range, milkyway_ebv):
        #This makes a new array of time which has 0.5 day cadence
        # times = np.linspace(mjd_range[0], mjd_range[1], self.mjd_bins) 
        times = np.arange(mjd_range[0], mjd_range[1], 1) 

        wavelengths = np.linspace(3000.0, 10100.0, self.wavelength_bins)
        ext = self._get_extinction(milkyway_ebv, wavelengths)
        ext = np.tile(np.expand_dims(ext, axis=1), len(times))

        time_wavelength_grid = np.transpose([np.tile(times, len(wavelengths)), np.repeat(wavelengths, len(times))])
        predictions, prediction_vars = gp(time_wavelength_grid, return_var=True)
        ext_corrected_predictions = np.array(predictions).reshape(len(wavelengths), len(times)) + ext
        prediction_uncertainties = np.sqrt(prediction_vars).reshape(len(wavelengths), len(times))

        return ext_corrected_predictions, prediction_uncertainties


    @staticmethod
    def _image_example(image_string, label, id):
        def _bytes_feature(value):
            """Returns a bytes_list from a string / byte."""
            if isinstance(value, type(tf.constant(0))):
                value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def _int64_feature(value):
            """Returns an int64_list from a bool / enum / int / uint."""
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        feature = {
            'id': _int64_feature(id),
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(image_string),
        }

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

