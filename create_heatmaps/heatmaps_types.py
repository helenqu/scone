import numpy as np
from create_heatmaps.base import CreateHeatmapsBase

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

class CreateHeatmapsEarlyMixed(CreateHeatmapsEarlyBase):
    def run(self):
        self.create_heatmaps([self.output_path], [[-20, [5,15,25,50]]], fit_on_full_lc=False)
    
    @staticmethod
    def _calculate_mjd_range(sn_metadata, sn_data, mjd_minmax, has_peakmjd):
        mjd_min, mjd_max = mjd_minmax
        trigger = self._calculate_trigger(sn_metadata, sn_data, mjd_minmax, has_peakmjd)
        if not trigger:
            return
        mjd_max = np.random.choice(mjd_max)
        return [trigger_mjd+mjd_min, trigger_mjd+mjd_max]

class CreateHeatmapsEarly(CreateHeatmapsEarlyBase):
    def run(self):
        days_after_trigger = [5,15,25,50]
        days_before_trigger = -20
        output_paths = [f"{self.output_path}/{days_before_trigger}x{i}_trigger" for i in days_after_trigger]
        mjd_ranges = [[days_before_trigger, i] for i in days_after_trigger]

        self.create_heatmaps(output_paths, mjd_ranges, fit_on_full_lc=False)
    
    @staticmethod
    def _calculate_mjd_range(sn_metadata, sn_data, mjd_minmax, has_peakmjd):
        mjd_min, mjd_max = mjd_minmax
        trigger = self._calculate_trigger(sn_metadata, sn_data, mjd_minmax, has_peakmjd)
        if not trigger:
            return
        return [trigger-mjd_min, trigger+mjd_max]

class CreateHeatmapsEarlyBase(CreateHeatmapsBase):
    def run(self):
        raise NotImplementedError

    @staticmethod
    def _calculate_mjd_range(sn_metadata, sn_data, mjd_minmax, has_peakmjd):
        raise NotImplementedError

    @staticmethod
    def _calculate_trigger(sn_metadata, sn_data, mjd_minmax, has_peakmjd):
        mjd_min, mjd_max = mjd_minmax
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

        return trigger_mjd

