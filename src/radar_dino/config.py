"""Versioned inference configuration for Radar-DINO model artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


def _pair(value: int | Sequence[int], name: str) -> tuple[int, int]:
    if isinstance(value, int):
        result = (value, value)
    else:
        result = tuple(int(item) for item in value)
        if len(result) != 2:
            raise ValueError(f"{name} must contain exactly two values")
    if any(item <= 0 for item in result):
        raise ValueError(f"{name} values must be positive")
    return result


@dataclass(frozen=True)
class RadarDINOConfig:
    """The preprocessing and architecture contract attached to one checkpoint."""

    model_id: str
    architecture: str
    fields: tuple[str, ...]
    normalization: Mapping[str, tuple[float, float]]
    patch_size: int = 5
    input_size: tuple[int, int] = (300, 300)
    positional_embedding_size: tuple[int, int] | None = None
    z_level_m: float | None = 2000.0
    nan_fill: float = -1.0
    weights_file: str = "model.safetensors"
    checkpoint_key: str | None = "teacher"
    spatial_dims: tuple[str, str] = ("y", "x")
    field_aliases: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    mask_with_reflectivity: bool = True
    schema_version: int = 1
    embedding_dimension: int | None = None
    grid_spacing_km: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "input_size", _pair(self.input_size, "input_size"))
        positional_embedding_size = (
            self.input_size
            if self.positional_embedding_size is None
            else _pair(self.positional_embedding_size, "positional_embedding_size")
        )
        object.__setattr__(
            self,
            "positional_embedding_size",
            positional_embedding_size,
        )
        object.__setattr__(self, "spatial_dims", tuple(self.spatial_dims))
        object.__setattr__(
            self,
            "normalization",
            {name: tuple(float(value) for value in limits) for name, limits in self.normalization.items()},
        )
        object.__setattr__(
            self,
            "field_aliases",
            {name: tuple(aliases) for name, aliases in self.field_aliases.items()},
        )
        self.validate()

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported manifest schema_version={self.schema_version}")
        if not self.model_id.strip():
            raise ValueError("model_id cannot be empty")
        if not self.fields:
            raise ValueError("At least one radar field is required")
        if len(set(self.fields)) != len(self.fields):
            raise ValueError("Radar fields must be unique")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if any(size % self.patch_size for size in self.input_size):
            raise ValueError("Every input_size dimension must be divisible by patch_size")
        if self.positional_embedding_size[0] != self.positional_embedding_size[1]:
            raise ValueError("positional_embedding_size must be square")
        if len(self.spatial_dims) != 2 or len(set(self.spatial_dims)) != 2:
            raise ValueError("spatial_dims must contain two unique dimension names")
        missing_normalization = set(self.fields) - set(self.normalization)
        if missing_normalization:
            missing = ", ".join(sorted(missing_normalization))
            raise ValueError(f"Missing fixed normalization limits for: {missing}")
        for name in self.fields:
            limits = self.normalization[name]
            if len(limits) != 2 or limits[0] >= limits[1]:
                raise ValueError(f"Invalid normalization limits for {name}: {limits}")
        unknown_aliases = set(self.field_aliases) - set(self.fields)
        if unknown_aliases:
            unknown = ", ".join(sorted(unknown_aliases))
            raise ValueError(f"Aliases declared for unknown fields: {unknown}")
        if self.grid_spacing_km is not None and self.grid_spacing_km <= 0:
            raise ValueError("grid_spacing_km must be positive")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "RadarDINOConfig":
        return cls(**dict(values))

    @classmethod
    def from_json(cls, path: str | Path) -> "RadarDINOConfig":
        manifest_path = Path(path)
        with manifest_path.open("r", encoding="utf-8") as stream:
            values = json.load(stream)
        if not isinstance(values, dict):
            raise ValueError(f"Expected a JSON object in {manifest_path}")
        return cls.from_dict(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "architecture": self.architecture,
            "embedding_dimension": self.embedding_dimension,
            "patch_size": self.patch_size,
            "fields": list(self.fields),
            "input_size": list(self.input_size),
            "positional_embedding_size": list(self.positional_embedding_size),
            "z_level_m": self.z_level_m,
            "nan_fill": self.nan_fill,
            "normalization": {name: list(limits) for name, limits in self.normalization.items()},
            "weights_file": self.weights_file,
            "checkpoint_key": self.checkpoint_key,
            "spatial_dims": list(self.spatial_dims),
            "field_aliases": {name: list(aliases) for name, aliases in self.field_aliases.items()},
            "mask_with_reflectivity": self.mask_with_reflectivity,
            "grid_spacing_km": self.grid_spacing_km,
        }

    def to_json(self, path: str | Path) -> None:
        manifest_path = Path(path)
        with manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=2)
            stream.write("\n")
