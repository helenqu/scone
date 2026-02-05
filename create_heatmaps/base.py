# base program for create_heatmaps
#
# Aug 22 2025: fix get_hdf5_ids_name() to be more robust and not rely on SNIaMODEL in FITS file name
# Sep 12 2025: remove self.LEGACY and self.REFAC .. keep only REFAC code
# Jan 29 2026: 
#   + include heatmap PROCESS_RATE (number per minute) in heatmap-summary files
#   + open START_TIME_STAMP.TXT file

import numpy as np
import pandas as pd
import tensorflow as tf

import os, sys, logging, yaml, argparse, h5py, abc, time

from datetime import datetime
from astropy.table import Table
from create_heatmaps.helpers import build_gp, image_example, get_extinction, read_fits, get_band_to_wave


SIMTAG_Ia    = "Ia"      # from GENTYPE_TO_CLASS dict in sim readme
SIMTAG_nonIa = "nonIa"

class CreateHeatmapsBase(abc.ABC):
    def __init__(self, config, index):
        self.index = index

        self.survey = config.get("survey", None)

        # file paths
        self.mode          = config['mode']
        self.metadata_path = config["metadata_paths"][index]
        self.lcdata_path   = config["lcdata_paths"][index]
        self.output_path   = config["heatmaps_path"]
        self.finished_filenames_path = os.path.join(self.output_path, "finished_filenames.csv")

        # heatmap parameters / metadata
        self.wavelength_bins = config["num_wavelength_bins"]
        self.mjd_bins        = config["num_mjd_bins"]
        self.has_peakmjd     = config.get("has_peakmjd", True)

        # load info from sim-readme that has been appended to config (4.3.2024, RK)
        self.SIM_GENTYPE_TO_CLASS = config.setdefault("SIM_GENTYPE_TO_CLASS",{}) 
        self.band_to_wave         = config.setdefault("band_to_wave", None) 

        self.IS_DATA_REAL = len(self.SIM_GENTYPE_TO_CLASS) == 0
        self.IS_DATA_SIM  = len(self.SIM_GENTYPE_TO_CLASS) >  0

        # - - - - - - - 
        # RK 4.2.2024: if type_to_name map is not already read from sim-data readme,
        #              then use legacy feature to read it from user config file.

        map_source       = "sim-readme (refac)"
        self.categorical = False  # disable for now; maybe restore later
        self.types       = [SIMTAG_nonIa, SIMTAG_Ia]
        self.type_to_int_label  = {SIMTAG_nonIa: 0,   SIMTAG_Ia: 1}
        self.sn_type_id_to_name = self.SIM_GENTYPE_TO_CLASS


        logging.info(f"")
        logging.info(f"TYPE <--> NAME maps from {map_source}:")
        logging.info(f"  List of types: {self.types}")
        logging.info(f"  sn_type_id_to_name: {self.sn_type_id_to_name}")
        logging.info(f"  type_to_int_label : {self.type_to_int_label}")

        # - - - - - - - -
        # restricting number of heatmaps that are made
        self.hdf5_select_file = config.get("hdf5_select_file", None)

        self.load_data()

        return


    def load_data(self):

        # Jan 14 2026 RK - few fixes to work with empty phot file and not crash; see n_lcdata

        logging.info(f'job {self.index:3d}: process file {self.lcdata_path}')

        if os.path.exists(self.finished_filenames_path):
            finished_filenames = pd.read_csv(self.finished_filenames_path)
            if os.path.basename(self.metadata_path) in finished_filenames:
                logging.info(" file has already been processed, exiting")
                sys.exit(0)
                
        self.metadata, self.lcdata, survey = \
                read_fits(self.lcdata_path, self.sn_type_id_to_name, self.survey)

        n_lcdata = len(self.lcdata)

        true_target  = None
        if self.IS_DATA_REAL:
            metadata_ids = self.metadata.object_id # take everything for real data
        else:
            metadata_ids = self.metadata[self.metadata.true_target.isin(self.types)].object_id
            if n_lcdata > 0 : true_target  = self.metadata.true_target.iloc[0] 

        self.lcdata.add_index('object_id')
        self.lcdata['passband'] = [flt.strip() for flt in self.lcdata['passband']]
        self.lcdata_ids  = np.intersect1d(self.lcdata['object_id'], metadata_ids)
        self.true_target = true_target  

        # survey info
        if self.band_to_wave is None:
            self.band_to_wave = get_band_to_wave(survey) # legacy : hard-wired params

        ids_path = self.hdf5_select_file      # refactored, RK

        if ids_path and n_lcdata > 0 :
            logging.info(f"Open snid-select file:  {ids_path}")
            ids_file     = h5py.File(ids_path, "r")
            ids_name     = self.get_hdf5_ids_name()
            self.ps_list = ids_file['prescales'][()]  # recover prescales used to select [Ia,nonIa]
            self.ids     = ids_file[ids_name][()]     # turn this into a numpy array
            n_ids        = len(self.ids)
            logging.info(f"  Select-Prescale list [Ia,nonIa] = {self.ps_list}")
            logging.info(f"  Example selected {ids_name}:  {self.ids[0:4]} ... of {n_ids}")
            ids_file.close()
        else:
            self.ps_list = [ 1, 1]
        
        if n_lcdata > 0:
            self.has_ids = ids_path and self.ids is not None
            self.ids_for_current_file = np.intersect1d(self.lcdata_ids, self.ids) \
                                        if self.has_ids else self.lcdata_ids
        else:
            self.has_ids = []
            self.ids_for_current_file = []

        logging.info(f"job {self.index:3d}: {'found' if self.has_ids else 'no'} idList, " \
                     f"expect {len(self.ids_for_current_file)}/{len(self.lcdata_ids)} heatmaps for this file")

        return
        # end load_data

    def get_hdf5_ids_name(self):

        # Apr 2024
        # Return name of snid set in hdf5 file.
        # For legacy and real data, name is "ids".
        # For sim, name is ids_Ia or ids_nonIa (to avoid random SNID overlaps)
        #
        # Aug 22 2025: fix logic to be more robust

        ids_base_name = "ids"

        if self.IS_DATA_REAL:
            ids_name = ids_base_name
        else:
            simtag = self.true_target
            ids_name = ids_base_name + '_'  + simtag        # ids_Ia or ids_nonIa

        return ids_name
    
    @abc.abstractmethod
    def run(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def _calculate_mjd_range(sn_metadata, sn_data, mjd_minmax, has_peakmjd):
        pass

    def create_heatmaps(self, output_paths, mjd_minmaxes, fit_on_full_lc=True):

        logging.info(f"tensorflow version: {tf.__version__}")

        #TODO: infer this from config file rather than making the subclasses pass it in
        self.fit_on_full_lc = fit_on_full_lc

        for output_path, mjd_minmax in zip(output_paths, mjd_minmaxes):
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            self.done_by_type    = {}
            self.removed_by_type = {}
            self.done_ids        = []

            timings = []
            self.t_start = time.time()  # need this here and to make summary file
            n_lc_read = 0  # RK start counter for diagnostic print
            n_lc_write = 0

            heatmap_file = f"{output_path}/heatmaps_{self.index:04d}.tfrecord"
            logging.info(f"job {self.index:3d}: write heatmaps to {heatmap_file}" )

            # create start-time stamp file to track wall time later in SCONE_SUMMARY
            time_stamp_file = f"{output_path}/START_TIME_STAMP.TXT"
            if not os.path.exists(time_stamp_file):
                with open(time_stamp_file,"wt") as t:
                    tnow = str( datetime.now() )
                    t.write(f"{tnow}\n")
                
            # - - - - -
            # begin writing heatmap
            with tf.io.TFRecordWriter(heatmap_file) as writer:
                for i, sn_id in enumerate(self.ids_for_current_file):
                    self.print_heatmap_status(i)
                    sn_name, *sn_data = self._get_sn_data(sn_id, mjd_minmax)

                    n_lc_read += 1                    
                    n_lc_write += 1

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

                    z     = sn_metadata['true_z'].iloc[0]
                    z_err = sn_metadata['true_z_err'].iloc[0]

                    if i == -99 :
                        print(f" xxx ------------------------------------- \n")
                        print(f" xxx i={i}  SNID={sn_name}  z={z:.3f} +_ {z_err:.3f}  " \
                              f" mwebv={milkyway_ebv:.4f}\n")
                        print(f" xxx wave = {wave}")
                        print(f" xxx sn_data = {sn_data}")
                        sys.stdout.flush() 
                        print(f" xxx heatmap = \n{heatmap}\n")  
                        sys.stdout.flush() 


                    writer.write(image_example(heatmap.flatten().tobytes(), 
                                               self.type_to_int_label[sn_name], sn_id, z, z_err))

                    
                    self._done(sn_name, sn_id)
                    
            # - - - -
            logging.info(f"job {self.index:3d}: Finished processing " \
                         f"{n_lc_write} of {n_lc_read} light curves.")

            # - - - - - 

            self.write_summary_file(heatmap_file) 
            return
            
    def print_heatmap_status(self,i):
        # Created Jun 2024 by R.Kessler
        # print status every 1000 events, and predict remaining time after 1st 1000 events.
        if i % 1000 == 0 :
            n_lc = len(self.ids_for_current_file)
            logging.info(f"job {self.index:3d}: processing {i} of {n_lc} light curves" )
            if i == 1000:
                time_to_1000  = (time.time() - self.t_start)/60.0  # minutes
                time_predict  = (len(self.ids_for_current_file)/1000)*time_to_1000
                msg_t = f"job {self.index}: {time_to_1000:.2f} min for 1000 heatmaps; " \
                        f"predict total process time: {time_predict:.2f} min"
                logging.info(f"{msg_t}")

        return
        
    # ================================================
    # HELPER FUNCTIONS
    # ================================================

    def remove_apply_prescale_reject(self, sn_id, sn_name):

        # Created Apr 2024 R.Kessler
        # check if event is rejected based on prescale and type (Ia,nonIa).
        # Return True to reject event.
        # Input sn_name is the general string type: either 'Ia' or 'nonIa'


        reject  = False
        ps_dict = self.prescale_heatmaps_dict

        if ps_dict is None  : 
            return reject  # never reject for predict mode

        ps = ps_dict[sn_name]
        if int(sn_id) % ps != 0:  
            reject = True

        return reject

    def write_summary_file(self, heatmap_file):
        
        # Created Mar 1 2024 by R.Kessler

        heatmap_file_base = os.path.basename(heatmap_file)

        # summary file name is heatmap file name with .tfrecord replaced by .summary
        summary_file = heatmap_file.split('.')[0] + '.summary'
        
        logging.info(f"Create summary file: {summary_file}")

        with open(summary_file,"wt") as s:
            s.write(f"PROGRAM_CLASS:  CreateHeatmaps\n")
            s.write(f"SURVEY:         {self.survey} \n")
            s.write(f"HEATMAP_FILE:   {heatmap_file_base}\n")
            s.write(f"JOBID:          {self.index}\n")
            s.write(f"LCDATA_PATH:    {self.lcdata_path} \n")
            s.write(f"PRESCALE_HEATMAPS:  {self.ps_list}\n")
            
            ntot_lc = 0
            s.write(f"N_LC: \n")
            if self.mode == 'train':
                for lctype, n_lc in self.done_by_type.items():
                    ntot_lc += n_lc
                    s.write(f"  {lctype}:  {n_lc:6d}       # TYPE: NLC\n")
            
            else:
                n_lc = len(self.done_ids)
                ntot_lc += n_lc
                s.write(f"  total:  {n_lc:6d} \n")

            # get process time and process rate for this bundle of heatmaps
            t_proc_sec      = (time.time() - self.t_start )
            t_proc_minutes  = t_proc_sec / 60.0
            proc_rate       = int(ntot_lc / t_proc_minutes)

            s.write(f"\n")
            s.write(f"CPU:            {t_proc_minutes:.2f}            # minutes \n")
            s.write(f"PROCESS_RATE:   {proc_rate:5d}     # <n_heatmap per minute> \n")

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

        # Jan 22 2026 RK 
        #  + if there is just one obs, sn_lcfilters is a scalar instead of list and causes crash.
        #    Simple fix is to return None if len(sn_lcfilters) <=1.

        sn_metadata = self.metadata[self.metadata.object_id == sn_id]
        if sn_metadata.empty:
            logging.info("sn metadata empty")
            return None, None

        sn_name      = sn_metadata.true_target.iloc[0]
        already_done = sn_id in self.done_ids
        if already_done:
            return sn_name, None

        sn_lcdata = self.lcdata.loc['object_id', sn_id]['mjd', 'flux', 'flux_err', 'passband']
        n_lcdata = len(sn_lcdata)
        if n_lcdata == 0 or np.all(sn_lcdata['mjd'] < 0):
            logging.info("Insufficient lcdata for sn_id={sn_id} : n_lcdata = {n_lcdata}")
            return sn_name, None

        expected_filters = list(self.band_to_wave.keys())  # valid list of filters
        sn_lcfilters     = sn_lcdata['passband'].tolist()  # lc filter list; incluces '-' for pad lc rows
        isin_filt_list   = np.isin(sn_lcfilters, expected_filters)

        if len(sn_lcfilters) <= 1 :            
            logging.info(f"Insufficient observations for sn_id={sn_id}: sn_lcfilters = {sn_lcfilters}")
            return sn_name, None

        # xxx mark 1.22.2026: sn_lcdata = sn_lcdata[np.isin(sn_lcdata['passband'], expected_filters)]
        sn_lcdata = sn_lcdata[isin_filt_list]

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

        # extend light curve to include very early & late epoch with zero flux.
        # Beware to pass flux_err > 0 to avoid divide-by-zero in build_gp.
        mjd_early = min(sn_lcdata['mjd']) - 100
        mjd_late  = max(sn_lcdata['mjd']) + 100
        flux = 0.0;  flux_err = 0.1;  band=expected_filters[2]
        sn_lcdata.add_row( [ mjd_early, flux, flux_err, band ] )
        sn_lcdata.add_row( [ mjd_late,  flux, flux_err, band ] )

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
