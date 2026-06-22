from __future__ import annotations

import numpy as np
import pytest
import torch
import xarray as xr

from utils import UnlabeledRadarNetCDFDataset


pytestmark = pytest.mark.unit


def test_dataset_discovers_nested_netcdf_and_returns_path(radar_netcdf):
    dataset = UnlabeledRadarNetCDFDataset(
        str(radar_netcdf.parents[1]),
        fields=("reflectivity",),
        z_level=2000.0,
        return_paths=True,
    )

    sample, label, path = dataset[0]

    assert len(dataset) == 1
    assert label == 0
    assert path == str(radar_netcdf)
    assert sample.shape == (1, 2, 3)


def test_dataset_defaults_to_2000_m(radar_netcdf):
    dataset = UnlabeledRadarNetCDFDataset(
        str(radar_netcdf),
        fields=("reflectivity",),
    )

    sample, _ = dataset[0]

    # At 2,000 m the first reflectivity value is 15 dBZ, normalized over 10-75.
    assert dataset.z_level == 2000.0
    assert sample[0, 0, 0].item() == pytest.approx(5.0 / 65.0)


def test_dataset_normalizes_channels_and_applies_reflectivity_mask(radar_netcdf):
    dataset = UnlabeledRadarNetCDFDataset(
        str(radar_netcdf),
        fields=("reflectivity", "differential_reflectivity"),
        z_level=2000.0,
        nan_fill=-1.0,
    )

    sample, _ = dataset[0]

    expected_reflectivity = torch.tensor(
        [
            [5.0 / 65.0, 15.0 / 65.0, 25.0 / 65.0],
            [35.0 / 65.0, 45.0 / 65.0, 55.0 / 65.0],
        ],
        dtype=torch.float32,
    )
    expected_zdr = torch.tensor(
        [
            [0.1, 0.3, 0.5],
            [0.7, 0.9, -1.0],
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(sample[0], expected_reflectivity)
    torch.testing.assert_close(sample[1], expected_zdr)


def test_column_max_is_selected_when_z_level_is_none(radar_netcdf):
    dataset = UnlabeledRadarNetCDFDataset(
        str(radar_netcdf),
        fields=("reflectivity",),
        z_level=None,
    )

    sample, _ = dataset[0]

    # The 2,000 m level is greater at every finite location in this fixture.
    assert sample.shape == (1, 2, 3)
    assert sample[0, 0, 0].item() == pytest.approx(5.0 / 65.0)


def test_missing_requested_field_raises_clear_error(radar_netcdf):
    dataset = UnlabeledRadarNetCDFDataset(
        str(radar_netcdf),
        fields=("spectrum_width",),
        z_level=2000.0,
    )

    with pytest.raises(KeyError, match="spectrum_width"):
        dataset[0]


def test_directory_without_netcdf_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Found no valid NetCDF files"):
        UnlabeledRadarNetCDFDataset(str(tmp_path))


def test_unknown_field_uses_data_driven_normalization(tmp_path):
    values = np.array([[[[2.0, 4.0], [6.0, np.nan]]]], dtype=np.float32)
    dataset = xr.Dataset(
        {"custom": (("time", "z", "y", "x"), values)},
        coords={"time": [0], "z": [1000.0], "y": [0.0, 1.0], "x": [0.0, 1.0]},
    )
    path = tmp_path / "custom.nc"
    dataset.to_netcdf(path, engine="scipy")

    radar_dataset = UnlabeledRadarNetCDFDataset(
        str(path),
        fields=("custom",),
        z_level=1000.0,
        nan_fill=-1.0,
    )
    sample, _ = radar_dataset[0]

    assert sample.min().item() == -1.0
    assert sample.max().item() == pytest.approx(1.0)

