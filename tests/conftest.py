from __future__ import annotations

import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def radar_netcdf(tmp_path):
    """Create a small deterministic time/z/y/x radar grid."""
    reflectivity = np.array(
        [
            [
                [[5.0, 10.0, 20.0], [30.0, np.nan, 80.0]],
                [[15.0, 25.0, 35.0], [45.0, 55.0, 65.0]],
            ]
        ],
        dtype=np.float32,
    )
    zdr = np.array(
        [
            [
                [[-8.0, -4.0, 0.0], [4.0, 8.0, 12.0]],
                [[-6.0, -2.0, 2.0], [6.0, 10.0, np.nan]],
            ]
        ],
        dtype=np.float32,
    )
    dataset = xr.Dataset(
        {
            "reflectivity": (("time", "z", "y", "x"), reflectivity),
            "differential_reflectivity": (("time", "z", "y", "x"), zdr),
        },
        coords={
            "time": [0],
            "z": [1000.0, 2000.0],
            "y": [0.0, 1.0],
            "x": [0.0, 1.0, 2.0],
        },
    )
    path = tmp_path / "nested" / "KHTX20250101_120000_V06.nc"
    path.parent.mkdir()
    dataset.to_netcdf(path, engine="scipy")
    return path

