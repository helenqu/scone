# base program for create_heatmaps

import numpy as np
import os, sys, logging
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

        self.survey = config.get("survey", None)

        # file paths
        self.metadata_path = config["metadata_paths"][index]
        self.lcdata_path   = config["lcdata_paths"][index]
        self.output_path   = config["heatmaps_path"]
        self.finished_filenames_path = os.path.join(self.output_path, "finished_filenames.csv")

        # heatmap parameters / metadata
        self.wavelength_bins = config["num_wavelength_bins"]
        self.mjd_bins        = config["num_mjd_bins"]
        self.has_peakmjd     = config.get("has_peakmjd", True)

        # heatmap labeling
        self.categorical = config["categorical"]
        self.types       = config["types"]
        logging.info(f"List of types: {self.types}")
        self.sn_type_id_to_name = config["sn_type_id_to_name"] # SNANA type ID to type name (i.e. 42 -> SNII)
        self.type_to_int_label = {type_str: 1 if type_str == "SNIa" or type_str == "Ia" else 0 for type_str in self.types} if not self.categorical else {v:i for i,v in enumerate(sorted(self.types))} # int label for classification

        logging.info(f"type to int label: {self.type_to_int_label}")

        # restricting number of heatmaps that are made
        self.ids_path = config.get("ids_path", None)

        # get sim fraction (Mar 1 2024)
        self.prescale_heatmap = config['prescale_heatmap']
        ps = self.prescale_heatmap
        if ps != 1 :
            logging.info(f"Select 1/{ps} of light curves\n")

        self.load_data()
        return

    def load_data(self):
        logging.info(f'job {self.index}: process file {self.lcdata_path}')

        if os.path.exists(self.finished_filenames_path):
            finished_filenames = pd.read_csv(self.finished_filenames_path)
            if os.path.basename(self.metadata_path) in finished_filenames:
                logging.info(" file has already been processed, exiting")
                sys.exit(0)

        self.metadata, self.lcdata, survey = read_fits(self.lcdata_path, self.sn_type_id_to_name, self.survey)
        metadata_ids = self.metadata[self.metadata.true_target.isin(self.types)].object_id

        self.lcdata.add_index('object_id')
        self.lcdata['passband'] = [flt.strip() for flt in self.lcdata['passband']]
        self.lcdata_ids = np.intersect1d(self.lcdata['object_id'], metadata_ids)

        # survey info
        self.band_to_wave = get_band_to_wave(survey)

        if self.ids_path:
            ids_file = h5py.File(self.ids_path, "r")
            self.ids = ids_file["ids"][()] # turn this into a numpy array
            logging.info(f"example id {self.ids[0]}")
            ids_file.close()
        self.has_ids = self.ids_path and self.ids is not None
        self.ids_for_current_file = np.intersect1d(self.lcdata_ids, self.ids) if self.has_ids else self.lcdata_ids

        logging.info(f"job {self.index}: {'found' if self.has_ids else 'no'} idList, " \
                     f"expect {len(self.ids_for_current_file)}/{len(self.lcdata_ids)} heatmaps for this file")

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
            logging.info(f"job {self.index}: writing to {output_path}" )

            self.done_by_type = {}
            self.removed_by_type = {}
            self.done_ids = []

            timings = []
            self.t_start = time.time()  # need this here and to make summary file
            n_lc_read = 0  # RK start counter for diagnostic print
            n_lc_write = 0

            heatmap_file = f"{output_path}/heatmaps_{self.index:04d}.tfrecord"
            with tf.io.TFRecordWriter(heatmap_file) as writer:
                for i, sn_id in enumerate(self.ids_for_current_file):
                    if i % 1000 == 0:
                        n_lc = len(self.ids_for_current_file)
                        logging.info(f"job {self.index}: processing {i} of {n_lc} light curves" )
                        if i == 1000:
                            time_to_1000  = (time.time() - self.t_start)/60.0  # minutes
                            time_predict  = (len(self.ids_for_current_file)/1000)*time_to_1000
                            msg_t = f"job {self.index}: {time_to_1000:.2f} min for 1000 heatmaps; " \
                                    f"predict total process time: {time_predict:.2f} min"
                            logging.info(f"{msg_t}")

                    # apply pre-scale here.
                    n_lc_read += 1
                    if int(sn_id) % self.prescale_heatmap != 0:  # RK - Mar 1 2024
                        continue

                    n_lc_write += 1
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
                            logging.info(f"{sn_name} not in SN_TYPE_ID_MAP?? stopping now")
                            break
                        self.type_to_int_label[sn_name] = 1 if sn_name == "SNIa" or sn_name == "Ia" else 0

                    z = sn_metadata['true_z'].iloc[0]
                    z_err = sn_metadata['true_z_err'].iloc[0]

                    writer.write(image_example(heatmap.flatten().tobytes(), 
                                               self.type_to_int_label[sn_name], sn_id, z, z_err))
                    self._done(sn_name, sn_id)
                    
            # - - - -
            logging.info(f"job {self.index}: Finsished processing " \
                         f"{n_lc_write} of {n_lc_read} light curves.")


            # xxxxxxxxx mark delete Mar 3 2024 xxxxxxxxx
            # RK - collisions reading/writing to same finished_filenames file is likely
            #   leading to rate (1-2%) of tasks crashing. This information is
            #    contained in the new heatmap*summary file per task, where there is
            #    no write collision.
            do_legacy_finished_filenames = False
            if do_legacy_finished_filenames:
                if not os.path.exists(self.finished_filenames_path):
                    pd.DataFrame({"filenames": [os.path.basename(self.metadata_path)]}).to_csv(self.finished_filenames_path, index=False)
                else:
                    finished_filenames = pd.read_csv(self.finished_filenames_path)
                    finished_filenames = pd.concat([finished_filenames, 
                                                    pd.DataFrame({"filenames": [os.path.basename(self.metadata_path)]})] )
                    finished_filenames.to_csv(self.finished_filenames_path, index=False)
            # xxxxxxxx end mark delete xxxxxxxxxxxx


            # - - - - - 
            # write REPORT information to done.log file :
            self.write_done_file_legacy(output_path)  # original/unformatted
            self.write_summary_file(heatmap_file)     # Mar 2024 RK - formatted summary 
            return
            
        
    # ================================================
    # HELPER FUNCTIONS
    # ================================================

    def write_summary_file(self, heatmap_file):
        
        # Created Mar 1 2024 by R.Kessler

        heatmap_file_base = os.path.basename(heatmap_file)

        # summary file name is heatmap file name with .tfrecord replaced by .summary
        summary_file = heatmap_file.split('.')[0] + '.summary'
        
        # get process time for this bundle of heatmaps.
        t_proc_minutes  = (time.time() - self.t_start )/60.0

        with open(summary_file,"wt") as s:
            s.write(f"PROGRAM_CLASS:  CreateHeatmaps\n")
            s.write(f"SURVEY:         {self.survey} \n")
            s.write(f"HEATMAP_FILE:   {heatmap_file_base}\n")
            s.write(f"JOBID:          {self.index}\n")
            s.write(f"CPU:            {t_proc_minutes:.2f}            # minutes \n")
            s.write(f"LCDATA_PATH:    {self.lcdata_path} \n")

            s.write(f"NLC: \n")
            for lctype, n_lc in self.done_by_type.items():
                s.write(f"  {lctype}:  {n_lc:6d}       # TYPE: NLC\n")

            # cpu ??
            #s.write(f"\n# xxx type_to_int_label = {self.type_to_int_label}\n")
            #s.write(f"\n# xxx done_by_type  = {self.done_by_type}\n")
        return

    def write_done_file_legacy(self, output_path):
        # Created Mar 2024 by R.Kessler
        # Write old-style (unformatted) done_legacy.log file as we transition
        # to newer formatted summary. 
        # This function should be removed after transition is complete.

        done_file = f"{output_path}/done_legacy.log" 
        with open(done_file, "a+") as f:
            f.write(f"####### JOB {self.index} REPORT #######\n")
            photfile_base = os.path.basename(self.lcdata_path)
            f.write(f"  base PHOT file name: {photfile_base}\n")  # added by RK

            tmp0 = self.type_to_int_label
            tmp1 = str(self.done_by_type).replace("'", "") 
            f.write(f"  type name mapping to integer label for classification: {tmp0}{tmp1}\n")

            total = 0
            for v in self.done_by_type.values():
                total += v
            f.write(f"  done: {total}\n" )

            str_removed = str(self.removed_by_type).replace("'", "")  # ?? RK what is this ??
            f.write(f"  removed: {str_removed}\n")
        return

    def _get_sn_data(self, sn_id, mjd_minmax):
        #TODO: find a better thing to early return
        sn_metadata = self.metadata[self.metadata.object_id == sn_id]
        if sn_metadata.empty:
            logging.info("sn metadata empty")
            return None, None

        sn_name = sn_metadata.true_target.iloc[0]
        already_done = sn_id in self.done_ids
        if already_done:
            return sn_name, None

        sn_lcdata = self.lcdata.loc['object_id', sn_id]['mjd', 'flux', 'flux_err', 'passband']
        if len(sn_lcdata) == 0 or np.all(sn_lcdata['mjd'] < 0):
            logging.info("sn lcdata empty")
            return sn_name, None

        expected_filters = list(self.band_to_wave.keys())
        sn_lcdata = sn_lcdata[np.isin(sn_lcdata['passband'], expected_filters)]
        if len(sn_lcdata) == 0:
            logging.info("expected filters filtering not working")
            return sn_name, None

        mjd_range = self._calculate_mjd_range(sn_metadata, sn_lcdata, mjd_minmax, self.has_peakmjd)
        if not mjd_range:
            logging.info("mjd range is none")
            return sn_name, None

        if not self.fit_on_full_lc:
            mjds = sn_lcdata['mjd']
            mask = np.logical_and(mjds >= mjd_range[0], mjds <= mjd_range[1])
            if not mask.any(): # if all false
                logging.info("empty sn data after mjd mask", mjd_range, np.min(mjds), np.max(mjds))
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
