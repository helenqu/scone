import numpy as np
import os, sys
import pandas as pd
from astropy.table import Table
import tensorflow as tf
import yaml
import argparse
import h5py
import time
import abc
from create_heatmaps.helpers import build_gp, image_example, get_extinction, read_fits, get_band_to_wave

class CreateHeatmapsBase(abc.ABC):
    def __init__(self, config, index):
        self.index = index

        # file paths 
        self.metadata_path = config["metadata_paths"][index]
        self.lcdata_path = config["lcdata_paths"][index]
        self.output_path = config["heatmaps_path"]
        self.finished_filenames_path = os.path.join(self.output_path, "finished_filenames.csv")

        # heatmap parameters / metadata
        self.wavelength_bins = config["num_wavelength_bins"]
        self.mjd_bins = config["num_mjd_bins"]
        self.has_peakmjd = config.get("has_peakmjd", True)

        # heatmap labeling
        self.categorical = config["categorical"]
        self.types = config["types"]
        self.sn_type_id_to_name = config["sn_type_id_to_name"] # SNANA type ID to type name (i.e. 42 -> SNII)
        self.type_to_int_label = {type: 0 if type != "SNIa" else 1 for type in self.types} if not self.categorical else {v:i for i,v in enumerate(sorted(self.types))} # int label for classification
        print(f"type to int label: {self.type_to_int_label}")

        # restricting number of heatmaps that are made
        self.ids_path = config.get("ids_path", None)

        self.load_data()

    def load_data(self):
        print(f'Processing file: {self.lcdata_path}')

        if os.path.exists(self.finished_filenames_path):
            finished_filenames = pd.read_csv(self.finished_filenames_path)
            if os.path.basename(self.metadata_path) in finished_filenames:
                print("file has already been processed, exiting")
                sys.exit(0)
        
        self.metadata, self.lcdata, survey = read_fits(self.lcdata_path, self.sn_type_id_to_name)
        metadata_ids = self.metadata[self.metadata.true_target.isin(self.types)].object_id

        self.lcdata.add_index('object_id')
        self.lcdata['passband'] = [flt.strip() for flt in self.lcdata['passband']]
        self.lcdata_ids = np.intersect1d(self.lcdata['object_id'], metadata_ids)

        # survey info
        self.band_to_wave = get_band_to_wave(survey)

        if self.ids_path:
            ids_file = h5py.File(self.ids_path, "r")
            self.ids = ids_file["ids"][()] # turn this into a numpy array
            print(f"example id {self.ids[0]}")
            ids_file.close()
        self.has_ids = self.ids_path and self.ids is not None
        self.ids_for_current_file = np.intersect1d(self.lcdata_ids, self.ids) if self.has_ids else self.lcdata_ids
        print(f"job {self.index}: {'found' if self.has_ids else 'no'} ids, expect {len(self.ids_for_current_file)}/{len(self.lcdata_ids)} heatmaps for this file", flush=True)

    @abc.abstractmethod
    def run(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def _calculate_mjd_range(sn_metadata, sn_data, mjd_minmax, has_peakmjd):
        pass

    def create_heatmaps(self, output_paths, mjd_minmaxes, fit_on_full_lc=True):
        #TODO: infer this from config file rather than making the subclasses pass it in
        self.fit_on_full_lc = fit_on_full_lc
            
        for output_path, mjd_minmax in zip(output_paths, mjd_minmaxes):
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            print("writing to {}".format(output_path), flush=True)

            self.done_by_type = {}
            self.removed_by_type = {}
            self.done_ids = []
            
            timings = []
            start = time.time()
            with tf.io.TFRecordWriter("{}/heatmaps_{}.tfrecord".format(output_path, self.index)) as writer:
                for i, sn_id in enumerate(self.ids_for_current_file):
                    if i % 1000 == 0:
                        print("job {}: processing {} of {}".format(self.index, i, len(self.ids_for_current_file)), flush=True)
                        if i == 1000:
                            time_to_1000 = time.time() - start
                            print(f"took {time_to_1000} sec for 1000 heatmaps; expected total time: {(len(self.ids_for_current_file)/1000)*time_to_1000} sec")
                    
                    sn_name, *sn_data = self._get_sn_data(sn_id, mjd_minmax)
                    if sn_data[0] is None:
                        self._remove(sn_name)
                        continue
                    sn_metadata, sn_lcdata, mjd_range = sn_data
                    wave = [self.band_to_wave[elem] for elem in sn_lcdata['passband']]
                
                    gp = build_gp(20, sn_lcdata, wave)
                    if gp == None:
                        self._remove(sn_name)
                        continue

                    milkyway_ebv = sn_metadata['mwebv'].iloc[0]

                    predictions, prediction_errs = self._get_predictions_heatmap(gp, mjd_range, milkyway_ebv)
                    heatmap = np.dstack((predictions, prediction_errs))

                    if sn_name not in self.type_to_int_label:
                        if self.categorical:
                            print(f"{sn_name} not in SN_TYPE_ID_MAP?? stopping now")
                            break
                        self.type_to_int_label[sn_name] = 1 if sn_name == "SNIa" or sn_name == "Ia" else 0

                    z = sn_metadata['true_z'].iloc[0]
                    z_err = sn_metadata['true_z_err'].iloc[0]
                    
                    writer.write(image_example(heatmap.flatten().tobytes(), self.type_to_int_label[sn_name], sn_id, z, z_err))
                    self._done(sn_name, sn_id)

            if not os.path.exists(self.finished_filenames_path):
                pd.DataFrame({"filenames": [os.path.basename(self.metadata_path)]}).to_csv(self.finished_filenames_path, index=False)
            else:
                finished_filenames = pd.read_csv(self.finished_filenames_path)
                finished_filenames.append({"filenames": os.path.basename(self.metadata_path)}, ignore_index=True).to_csv(self.finished_filenames_path, index=False)

            with open("{}/done.log".format(output_path), "a+") as f:
                f.write("####### JOB {} REPORT #######\n".format(self.index))
                f.write("type name mapping to integer label used for classification: {}".format(self.type_to_int_label))
                f.write(str(self.done_by_type).replace("'", "") + "\n")
                total = 0
                for v in self.done_by_type.values():
                    total += v
                f.write("done: {}\n".format(total))
                f.write("removed: {}\n".format(str(self.removed_by_type).replace("'", "")))

    # HELPER FUNCTIONS
    def _get_sn_data(self, sn_id, mjd_minmax):
        #TODO: find a better thing to early return
        sn_metadata = self.metadata[self.metadata.object_id == sn_id]
        if sn_metadata.empty:
            print("sn metadata empty")
            return None, None

        sn_name = sn_metadata.true_target.iloc[0]
        already_done = sn_id in self.done_ids
        if already_done:
            return sn_name, None

        sn_lcdata = self.lcdata.loc['object_id', sn_id]['mjd', 'flux', 'flux_err', 'passband']

        expected_filters = list(self.band_to_wave.keys())
        sn_lcdata = sn_lcdata[np.isin(sn_lcdata['passband'], expected_filters)] 
        if len(sn_lcdata) == 0: 
            print("expected filters filtering not working") 
            return sn_name, None

        mjd_range = self._calculate_mjd_range(sn_metadata, sn_lcdata, mjd_minmax, self.has_peakmjd)
        if not mjd_range:
            print("mjd range is none") 
            return sn_name, None

        if not self.fit_on_full_lc:
            mjds = sn_lcdata['mjd']
            mask = np.logical_and(mjds >= mjd_range[0], mjds <= mjd_range[1])
            if not mask.any(): # if all false
                print("empty sn data after mjd mask", mjd_range, np.min(mjds), np.max(mjds))
                return sn_name, None
            sn_lcdata = sn_lcdata[mask]

        sn_lcdata.add_row([min(sn_lcdata['mjd'])-100, 0, 0, expected_filters[2]])
        sn_lcdata.add_row([max(sn_lcdata['mjd'])+100, 0, 0, expected_filters[2]])

        return sn_name, sn_metadata, sn_lcdata, mjd_range

    def _remove(self, sn_name):
        self.removed_by_type[sn_name] = 1 if sn_name not in self.removed_by_type else self.removed_by_type[sn_name] + 1

    def _done(self, sn_name, sn_id):
        self.done_ids.append(sn_id)
        self.done_by_type[sn_name] = 1 if sn_name not in self.done_by_type else self.done_by_type[sn_name] + 1

    def _get_predictions_heatmap(self, gp, mjd_range, milkyway_ebv):
        times = np.linspace(mjd_range[0], mjd_range[1], self.mjd_bins)

        wavelengths = np.linspace(3000.0, 10100.0, self.wavelength_bins)
        ext = get_extinction(milkyway_ebv, wavelengths)
        ext = np.tile(np.expand_dims(ext, axis=1), len(times))
        time_wavelength_grid = np.transpose([np.tile(times, len(wavelengths)), np.repeat(wavelengths, len(times))])
 
        predictions, prediction_vars = gp(time_wavelength_grid, return_var=True)
        ext_corrected_predictions = np.array(predictions).reshape(32, 180) + ext
        prediction_uncertainties = np.sqrt(prediction_vars).reshape(32, 180)

        return ext_corrected_predictions, prediction_uncertainties
