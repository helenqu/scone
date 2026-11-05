from types import SimpleNamespace

import numpy as np
import pytest

from create_heatmaps.base import CreateHeatmapsBase
from create_heatmaps.helpers import get_extinction

MEAN = 100.0
VARIANCE = 4.0
WAVELENGTHS = np.linspace(3000.0, 10100.0, 32)


def stub_gp(grid, return_var=True):
    return np.full(len(grid), MEAN), np.full(len(grid), VARIANCE)


def predict(ebv, legacy_extinction):
    heatmaps = SimpleNamespace(mjd_bins=180, wavelength_bins=32, legacy_extinction=legacy_extinction)
    return CreateHeatmapsBase._get_predictions_heatmap(heatmaps, stub_gp, (0.0, 179.0), ebv)


def test_extinction_scales_flux_and_uncertainty():
    mean, std = predict(0.05, legacy_extinction=False)
    factor = 10 ** (0.4 * get_extinction(0.05, WAVELENGTHS))[:, None]
    assert np.all(factor > 1)
    np.testing.assert_allclose(mean, MEAN * factor * np.ones((32, 180)))
    np.testing.assert_allclose(std, np.sqrt(VARIANCE) * factor * np.ones((32, 180)))


def test_legacy_extinction_adds_to_flux():
    mean, std = predict(0.05, legacy_extinction=True)
    extinction = get_extinction(0.05, WAVELENGTHS)[:, None]
    np.testing.assert_allclose(mean, MEAN + extinction * np.ones((32, 180)))
    np.testing.assert_allclose(std, np.full((32, 180), np.sqrt(VARIANCE)))


@pytest.mark.parametrize("legacy_extinction", [False, True])
def test_no_extinction_leaves_predictions_unchanged(legacy_extinction):
    mean, std = predict(0.0, legacy_extinction)
    np.testing.assert_allclose(mean, np.full((32, 180), MEAN))
    np.testing.assert_allclose(std, np.full((32, 180), np.sqrt(VARIANCE)))
