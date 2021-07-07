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

class CreateHeatmapsEarlyMixed(CreateHeatmapsEarlyBase):
    def run(self):
        self.create_heatmaps([self.output_path], [[-20, [5,15,25,50]]], fit_on_full_lc=False)
    
    @staticmethod
    def _calculate_mjd_range(sn_metadata, sn_data, mjd_minmax, has_peakmjd):
        mjd_min, mjd_max = mjd_minmax
        trigger = self._calculate_trigger(sn_metadata, sn_data)
        if not trigger:
            return
        mjd_max = np.random.choice(mjd_max)
        return [trigger_mjd+mjd_min, trigger_mjd+mjd_max]

class CreateHeatmapsEarly(CreateHeatmapsEarlyBase):
    def run(self):
        days_after_trigger = [5,15,25,50]
        days_before_trigger = -20
        output_paths = [f"{self.output_path}/{days_before_trigger}x{i}_trigger_32x180" for i in days_after_trigger]
        mjd_ranges = [[days_before_trigger, i] for i in days_after_trigger]

        self.create_heatmaps(output_paths, mjd_ranges, fit_on_full_lc=False)
    
    @staticmethod
    def _calculate_mjd_range(sn_metadata, sn_data, mjd_minmax, has_peakmjd):
        mjd_min, mjd_max = mjd_minmax
        trigger = self._calculate_trigger(sn_metadata, sn_data)
        if not trigger:
            return
        return [trigger-mjd_min, trigger+mjd_max]

class LastSNRById(CreateHeatmapsBase):
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
        last_snr_by_id = {5: {}, 15: {}, 25: {}, 50: {}}
        # num_detection_points_by_type = {5: {}, 15: {}, 25: {}, 50: {}}
        metadata_path = self.config['metadata_paths'][self.index]
        lcdata_path = self.config['lcdata_paths'][self.index]

        metadata = pd.read_csv(metadata_path, compression="gzip") if os.path.splitext(metadata_path)[1] == ".gz" else pd.read_csv(metadata_path)
        metadata_ids = metadata[metadata.true_target.isin(self.config["sn_type_id_to_name"].keys())].object_id

        lcdata = pd.read_csv(lcdata_path, compression="gzip") if os.path.splitext(lcdata_path)[1] == ".gz" else pd.read_csv(lcdata_path)
        lcdata = Table.from_pandas(lcdata)
        lcdata.add_index('object_id')
        lcdata_ids = np.intersect1d(lcdata['object_id'], metadata_ids)

        for i, sn_id in enumerate(lcdata_ids):
            if i % 1000 == 0:
                print(f"processing {i} of {len(lcdata_ids)}")
            sn_metadata = metadata[metadata.object_id == sn_id]
            sn_name = self.config["sn_type_id_to_name"][sn_metadata.true_target.iloc[0]]
            sn_lcdata = lcdata.loc['object_id', sn_id]['mjd', 'flux', 'flux_err', 'passband']

            for mjdmax in [5,15,25,50]:
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
                # sn_lcdata_included = sn_lcdata[mask]

                detections_mjd, detections_snr = detections
                mask = np.logical_and(detections_mjd >= mjd_range[0], detections_mjd <= mjd_range[1])
                detections_included = np.array(detections_snr)[mask]
                last_snr = detections_included[-1]

                last_snr_by_id[mjdmax][int(sn_id)] = last_snr

            with open(self.config["heatmaps_path"] + f"/last_snr_by_id_{self.index}.json", "w+") as outfile:
                outfile.write(json.dumps(last_snr_by_id))
