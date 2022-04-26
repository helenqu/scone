import numpy as np
import os
from create_heatmaps.base import CreateHeatmapsBase
import json
import pandas as pd

class CreateHeatmapsFull(CreateHeatmapsBase):
    def run(self):
        self.create_heatmaps([self.output_path], [[-30, 150]])

    @staticmethod
    def _calculate_mjd_range(sn_metadata, sn_data, mjd_minmax, has_peakmjd):
        mjd_min, mjd_max = mjd_minmax
        if not has_peakmjd:
            mjd_range = [np.min(sn_data['mjd']), np.max(sn_data['mjd'])]
            return mjd_range
        peak_mjd = sn_metadata['true_peakmjd'].iloc[0]
        mjd_range = [peak_mjd+mjd_min, peak_mjd+mjd_max]

        return mjd_range

class CreateHeatmapsEarlyBase(CreateHeatmapsBase):
    def run(self):
        raise NotImplementedError

    @staticmethod
    def _calculate_mjd_range(sn_metadata, sn_data, mjd_minmax, has_peakmjd):
        raise NotImplementedError

    @staticmethod
    def _calculate_trigger(sn_metadata, sn_data):
        sn_data.sort("mjd")
        snrs_by_mjd = [[mjd, flux/flux_err] for mjd, flux, flux_err in sn_data.iterrows('mjd', 'flux', 'flux_err')]
        detections = [[mjd,snr] for mjd, snr in snrs_by_mjd if snr > 5]
        if len(detections) < 2:
            return 
        first_detection_mjd = detections[0][0]
        # find first detection that occurred more than 1 day after initial detection
        detections = [detection for detection in detections if detection[0] > 1+first_detection_mjd]
        if len(detections) == 0:
            return
        trigger_mjd = detections[0][0]

        return trigger_mjd

class CreateHeatmapsEarlyMixed(CreateHeatmapsEarlyBase):
    def run(self):
        print("running early mixed")
        self.create_heatmaps([self.output_path], [[-20, np.arange(0,51)]], fit_on_full_lc=False)
    
    @staticmethod
    def _calculate_mjd_range(sn_metadata, sn_data, mjd_minmax, has_peakmjd):
        mjd_min, mjd_max = mjd_minmax
        trigger = CreateHeatmapsEarlyMixed._calculate_trigger(sn_metadata, sn_data)
        if not trigger:
            return
        mjd_max = np.random.choice(mjd_max)
        return [trigger+mjd_min, trigger+mjd_max],mjd_max #TODO: change backto one return val 7/16

class CreateHeatmapsEarly(CreateHeatmapsEarlyBase):
    def run(self):
        days_after_trigger = [150]
        days_before_trigger = -30
        output_paths = [f"{self.output_path}/{days_before_trigger}x{i}_trigger_32x180" for i in days_after_trigger]
        mjd_ranges = [[days_before_trigger, i] for i in days_after_trigger]

        self.create_heatmaps(output_paths, mjd_ranges, fit_on_full_lc=False)
    
    @staticmethod
    def _calculate_mjd_range(sn_metadata, sn_data, mjd_minmax, has_peakmjd):
        mjd_min, mjd_max = mjd_minmax
        trigger = CreateHeatmapsEarly._calculate_trigger(sn_metadata, sn_data)
        if not trigger:
            return
        return [trigger+mjd_min, trigger+mjd_max]

class SaveTriggerToCSV(CreateHeatmapsEarlyBase):
    def run(self):
        OUTPUT_PATH = os.path.dirname(self.metadata_path)
        print("writing to {}".format(OUTPUT_PATH))
        self.metadata["1season_peakmjd"] = np.zeros(len(self.metadata))
        self.metadata["3season_peakmjd"] = np.zeros(len(self.metadata))

        for i, sn_id in enumerate(self.lcdata_ids):
            if i % 1000 == 0:
                print(f"processing {i} of {len(self.lcdata_ids)}")
            sn_metadata = self.metadata[self.metadata.object_id == sn_id]
            sn_name = self.sn_type_id_map[sn_metadata.true_target.iloc[0]]
            sn_lcdata = self.lcdata.loc['object_id', sn_id]['mjd', 'flux', 'flux_err', 'passband']

            sn_mjd = sorted(sn_lcdata['mjd'], reverse=True)
            trigger = sn_metadata.trigger_mjd.values[0]
            if np.isnan(trigger):
                continue
            sn_mjd_trigger_idx = np.where([round(mjd, 3) for mjd in sn_mjd] == round(trigger, 3))[0]
            if len(sn_mjd_trigger_idx) == 0:
                print(trigger)
                print([round(mjd, 3) for mjd in sn_mjd])
                break
            sn_mjd_trigger_idx = sn_mjd_trigger_idx[0]

            season_start_idx = -1
            for i in range(sn_mjd_trigger_idx, len(sn_mjd)-1):
                if i == 0:
                    print(sn_mjd[i] - sn_mjd[i+1])
                if sn_mjd[i] - sn_mjd[i+1] > 50:
                    season_start_idx = i
                    break
            season_start = sn_mjd[season_start_idx]

            sn_mjd = sorted(sn_lcdata['mjd'])
            sn_mjd_trigger_idx = np.where([round(mjd, 3) for mjd in sn_mjd] == round(trigger, 3))[0][0]
            season_end_idx = -1
            for i in range(sn_mjd_trigger_idx, len(sn_mjd)-1):
                if sn_mjd[i] - sn_mjd[i+1] < -100:
                    season_end_idx = i
                    break
            season_end = sn_mjd[season_end_idx]

            sn_data = sn_lcdata[np.logical_and(sn_lcdata["mjd"] >= season_start, sn_lcdata["mjd"] <= season_end)]
            mjd = np.array(sn_data['mjd'])
            flux = np.array(sn_data['flux'])
            flux_err = np.array(sn_data['flux_err'])
            snrs = flux**2 / flux_err**2
            mask = snrs > 5
            mjd = mjd[mask]
            snrs = snrs[mask]
            peak_mjd_oneseason = np.sum(mjd * snrs) / np.sum(snrs)
            self.metadata.loc[self.metadata.object_id == sn_id, "1season_peakmjd"] = peak_mjd_oneseason

            mjd = np.array(sn_lcdata['mjd'])
            flux = np.array(sn_lcdata['flux'])
            flux_err = np.array(sn_lcdata['flux_err'])
            snrs = flux**2 / flux_err**2
            mask = snrs > 5
            mjd = mjd[mask]
            snrs = snrs[mask]
            if len(mjd) == 0 or len(snrs) == 0:
                print(snid)
            peak_mjd_calculated = np.sum(mjd * snrs) / np.sum(snrs)
            self.metadata.loc[self.metadata.object_id == sn_id, "3season_peakmjd"] = peak_mjd_calculated
        self.metadata.to_csv(os.path.join(OUTPUT_PATH, os.path.basename(self.metadata_path)), index=False)

class MagById(CreateHeatmapsBase):
    @staticmethod
    def _calculate_mjd_range(sn_metadata, sn_data, mjd_minmax, has_peakmjd):
        raise NotImplementedError
    def run(self):
        def _calculate_detections(sn_data):
            sn_data.sort("mjd")
            snrs_by_mjd = [[mjd, flux/flux_err] for mjd, flux, flux_err in sn_data.iterrows('mjd', 'flux', 'flux_err')]
            # snrs_by_mjd = {sn_data.iloc[idx]['mjd']:sn_data.iloc[idx]['flux']/sn_data.iloc[idx]['flux_err'] for idx in range(len(sn_data))}
            detections = [[mjd,snr] for mjd, snr in snrs_by_mjd if snr > 5]
            if len(detections) < 2:
                return 
            return [detection[0] for detection in detections], [detection[1] for detection in detections]

        def _calculate_trigger(sn_data):
            detections = _calculate_detections(sn_data)
            if not detections:
                return
            detections_mjd = detections[0]
            first_detection_mjd = detections_mjd[0]
            # find first detection that occurred more than 1 day after initial detection
            detections_mjd = [detection for detection in detections_mjd if detection > 1+first_detection_mjd]
            if len(detections_mjd) == 0:
                return
            trigger_mjd = detections_mjd[0]

            return trigger_mjd

        mag_by_id = {0: [], 5: [], 15: []}
        for i, sn_id in enumerate(self.lcdata_ids):
            if i % 1000 == 0:
                print(f"processing {i} of {len(self.lcdata_ids)}")
            sn_metadata = self.metadata[self.metadata.object_id == sn_id]
            sn_name = self.sn_type_id_map[sn_metadata.true_target.iloc[0]]
            sn_lcdata = self.lcdata.loc['object_id', sn_id]['mjd', 'flux', 'flux_err', 'passband']

            for mjdmax in mag_by_id.keys():
                trigger_mjd = _calculate_trigger(sn_lcdata)
                detections = _calculate_detections(sn_lcdata)
                if not detections or not trigger_mjd:
                    continue
                mjd_range = [trigger_mjd-20, trigger_mjd+mjdmax]

                mjds = sn_lcdata['mjd']
                mask = np.logical_and(mjds >= mjd_range[0], mjds <= mjd_range[1])
                if not mask.any(): # if all false
                    print("empty sn data after mjd mask", mjd_range, np.min(mjds), np.max(mjds))
                    continue
                sn_lcdata_included = sn_lcdata[mask]
                sn_lcdata_r = sn_lcdata_included[sn_lcdata_included['passband'] == 1]
                if len(sn_lcdata_r) == 0:
                    continue
                last_r_flux = sn_lcdata_r['flux'][-1]
                last_r_mag = 27.5 - 2.5*np.log10(last_r_flux)
                if last_r_mag <= 20:
                    mag_by_id[int(mjdmax)].append(int(sn_id))

        with open(os.path.join(self.output_path, f"mag_over_20_ids_{self.index}.json"), "w+") as outfile:
            json.dump(mag_by_id, outfile)

class SaveFirstDetectionToCSV(CreateHeatmapsEarlyBase):
    @staticmethod
    def _calculate_first_detection(sn_metadata, sn_data):
        sn_data.sort("mjd")
        snrs_by_mjd = [[mjd, flux/flux_err] for mjd, flux, flux_err in sn_data.iterrows('mjd', 'flux', 'flux_err')]
        detections = [[mjd,snr] for mjd, snr in snrs_by_mjd if snr > 5]
        if len(detections) < 2:
            return 
        first_detection_mjd = detections[0][0]

        return first_detection_mjd

    def run(self):
        OUTPUT_PATH = os.path.dirname(self.metadata_path)
        print("writing to {}".format(OUTPUT_PATH))

        data = []
        for i, sn_id in enumerate(self.lcdata_ids):
            if i % 1000 == 0:
                print(f"processing {i} of {len(self.lcdata_ids)}")
            sn_metadata = self.metadata[self.metadata.object_id == sn_id]
            sn_name = self.sn_type_id_map[sn_metadata.true_target.iloc[0]]
            sn_lcdata = self.lcdata.loc['object_id', sn_id]['mjd', 'flux', 'flux_err', 'passband']

            sn_mjd = sorted(sn_lcdata['mjd'], reverse=True)
            trigger = sn_metadata.trigger_mjd.values[0]
            first_detection = SaveFirstDetectionToCSV._calculate_first_detection(sn_metadata, sn_lcdata)
            if np.isnan(trigger) or np.isnan(first_detection):
                continue
            data.append([sn_id, first_detection, trigger])

        pd.DataFrame(data, columns=["snid", "first_detection_mjd", "trigger_mjd"]).to_csv(os.path.join(OUTPUT_PATH, os.path.basename(self.metadata_path)), index=False)
