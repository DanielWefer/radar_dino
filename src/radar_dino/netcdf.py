"""Strict NetCDF loading using the preprocessing contract in a model manifest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from .config import RadarDINOConfig


@dataclass(frozen=True)
class RadarSample:
    tensor: torch.Tensor
    path: Path
    fields: tuple[str, ...]
    source_fields: tuple[str, ...]
    original_shape: tuple[int, int]


def _requested_fields(
    requested: Iterable[str] | None,
    expected: tuple[str, ...],
) -> tuple[str, ...]:
    if requested is None:
        return expected
    requested_tuple = tuple(requested)
    if len(requested_tuple) != len(set(requested_tuple)):
        raise ValueError("Requested radar fields must be unique")
    if set(requested_tuple) != set(expected):
        missing = sorted(set(expected) - set(requested_tuple))
        extra = sorted(set(requested_tuple) - set(expected))
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"unexpected={extra}")
        raise ValueError(
            "Requested fields do not match the model contract: " + ", ".join(details)
        )
    return expected


def _source_name(dataset, canonical_name: str, config: RadarDINOConfig) -> str:
    candidates = (canonical_name, *config.field_aliases.get(canonical_name, ()))
    matches = [candidate for candidate in candidates if candidate in dataset.data_vars]
    if not matches:
        raise KeyError(
            f"NetCDF is missing required field '{canonical_name}'. "
            f"Accepted variable names: {', '.join(candidates)}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"NetCDF contains multiple aliases for '{canonical_name}': {matches}"
        )
    return matches[0]


def _select_slice(data, config: RadarDINOConfig):
    if "time" in data.dims:
        if data.sizes["time"] != 1:
            raise ValueError(
                f"Expected one time per NetCDF file, found {data.sizes['time']}"
            )
        data = data.isel(time=0)
    if "z" in data.dims:
        if config.z_level_m is None:
            data = data.max(dim="z", skipna=True)
        else:
            data = data.sel(z=config.z_level_m, method="nearest")

    extra_dims = [
        name
        for name in data.dims
        if name not in config.spatial_dims and data.sizes[name] != 1
    ]
    if extra_dims:
        raise ValueError(
            f"Radar field has unsupported non-spatial dimensions: {extra_dims}"
        )
    if any(
        name not in config.spatial_dims and data.sizes[name] == 1
        for name in data.dims
    ):
        data = data.squeeze(drop=True)
    if set(data.dims) != set(config.spatial_dims):
        raise ValueError(
            f"Expected spatial dimensions {config.spatial_dims}, got {data.dims}"
        )
    return data.transpose(*config.spatial_dims)


def _normalize(values: np.ndarray, limits: tuple[float, float], nan_fill: float) -> np.ndarray:
    low, high = limits
    missing = ~np.isfinite(values)
    normalized = np.nan_to_num(values, nan=low, posinf=high, neginf=low)
    normalized = np.clip(normalized, low, high)
    normalized = (normalized - low) / (high - low)
    normalized[missing] = nan_fill
    return normalized.astype(np.float32, copy=False)


def _center_crop(values: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    height, width = values.shape
    target_height, target_width = target
    if height < target_height or width < target_width:
        raise ValueError(
            f"Radar grid {values.shape} is smaller than required input size {target}"
        )
    top = (height - target_height) // 2
    left = (width - target_width) // 2
    return values[top : top + target_height, left : left + target_width]


def _validate_grid_spacing(dataset, config: RadarDINOConfig) -> None:
    if config.grid_spacing_km is None:
        return
    measured = []
    for dimension in config.spatial_dims:
        if dimension not in dataset.coords or dataset.coords[dimension].size < 2:
            raise ValueError(
                f"Cannot validate grid spacing because coordinate '{dimension}' is missing"
            )
        coordinate = dataset.coords[dimension]
        spacing = float(np.median(np.abs(np.diff(coordinate.values.astype(float)))))
        units = str(coordinate.attrs.get("units", "")).strip().lower()
        if units in {"m", "meter", "meters", "metre", "metres"}:
            spacing /= 1000.0
        elif units not in {"km", "kilometer", "kilometers", "kilometre", "kilometres"}:
            raise ValueError(
                f"Cannot interpret units {units!r} for coordinate '{dimension}'; "
                "expected meters or kilometers"
            )
        measured.append(spacing)
    if not all(
        np.isclose(value, config.grid_spacing_km, rtol=0.01, atol=1e-6)
        for value in measured
    ):
        raise ValueError(
            f"Grid spacing {tuple(measured)} km does not match model requirement "
            f"{config.grid_spacing_km} km"
        )


def load_radar_netcdf(
    path: str | Path,
    config: RadarDINOConfig,
    *,
    fields: Sequence[str] | None = None,
) -> RadarSample:
    """Load one radar grid, validate its fields, and return model-ready channels."""

    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError(
            "Reading NetCDF radar files requires xarray and a NetCDF backend."
        ) from exc

    radar_path = Path(path)
    if not radar_path.is_file():
        raise FileNotFoundError(f"Radar NetCDF file does not exist: {radar_path}")
    if radar_path.suffix.lower() not in {".nc", ".netcdf"}:
        raise ValueError(f"Expected a .nc or .netcdf file, got {radar_path.name}")

    canonical_fields = _requested_fields(fields, config.fields)
    channels: list[torch.Tensor] = []
    source_fields: list[str] = []
    original_shape: tuple[int, int] | None = None

    with xr.open_dataset(radar_path) as dataset:
        _validate_grid_spacing(dataset, config)
        resolved = {
            field: _source_name(dataset, field, config)
            for field in canonical_fields
        }
        reflectivity_mask = None
        if config.mask_with_reflectivity and "reflectivity" in resolved:
            reflectivity = _select_slice(dataset[resolved["reflectivity"]], config)
            reflectivity_values = reflectivity.values.astype(np.float32)
            low, high = config.normalization["reflectivity"]
            reflectivity_mask = (
                np.isfinite(reflectivity_values)
                & (reflectivity_values >= low)
                & (reflectivity_values <= high)
            )

        for field in canonical_fields:
            source_field = resolved[field]
            data = _select_slice(dataset[source_field], config)
            values = data.values.astype(np.float32)
            if original_shape is None:
                original_shape = tuple(values.shape)
            elif tuple(values.shape) != original_shape:
                raise ValueError(
                    f"Field '{source_field}' has shape {values.shape}; "
                    f"expected {original_shape}"
                )
            values = _normalize(values, config.normalization[field], config.nan_fill)
            if reflectivity_mask is not None:
                values[~reflectivity_mask] = config.nan_fill
            values = _center_crop(values, config.input_size)
            channels.append(torch.from_numpy(values.copy()))
            source_fields.append(source_field)

    assert original_shape is not None
    return RadarSample(
        tensor=torch.stack(channels, dim=0),
        path=radar_path.resolve(),
        fields=canonical_fields,
        source_fields=tuple(source_fields),
        original_shape=original_shape,
    )
