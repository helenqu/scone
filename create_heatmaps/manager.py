from create_heatmaps.heatmaps_types import CreateHeatmapsFull, CreateHeatmapsEarlyMixed, CreateHeatmapsEarly, MagById, SaveFirstDetectionToCSV

class CreateHeatmapsManager():
    def run(self, config, index):
        create_heatmaps_object = None
        if config.get("early_lightcurves_mixed", False):
            create_heatmaps_object = CreateHeatmapsEarlyMixed(config, index)
        elif config.get("early_lightcurves", False):
            create_heatmaps_object = CreateHeatmapsEarly(config, index)
        else:
            create_heatmaps_object = CreateHeatmapsFull(config, index)
        create_heatmaps_object.run()
