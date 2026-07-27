"""Typed inference results returned by the public Radar-DINO API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RadarDINOResult:
    path: Path
    fields: tuple[str, ...]
    feature: np.ndarray
    attention: np.ndarray | None = None
    umap: np.ndarray | None = None
    cluster: int | None = None
    cluster_probability: float | None = None
    neighbors: tuple[dict, ...] = ()

    def attention_for(self, field: str, head: int | str = "mean") -> np.ndarray:
        if self.attention is None:
            raise ValueError("Attention was not requested for this result")
        try:
            field_index = self.fields.index(field)
        except ValueError as exc:
            raise KeyError(f"Unknown radar field '{field}'") from exc
        field_attention = self.attention[:, field_index]
        if head == "mean":
            return field_attention.mean(axis=0)
        if not isinstance(head, int):
            raise TypeError("head must be an integer or 'mean'")
        return field_attention[head]
