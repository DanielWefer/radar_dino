"""Radar-DINO feature and field-attention inference."""

from .artifact import export_checkpoint
from .catalog import ReferenceCatalog
from .config import RadarDINOConfig
from .model import RadarDINO
from .plotting import (
    chasespectral_colormap,
    plot_attention,
    plot_attention_heads,
    plot_projection,
    save_analysis_plots,
)
from .result import RadarDINOResult

__all__ = [
    "RadarDINO",
    "RadarDINOConfig",
    "RadarDINOResult",
    "ReferenceCatalog",
    "export_checkpoint",
    "chasespectral_colormap",
    "plot_attention",
    "plot_attention_heads",
    "plot_projection",
    "save_analysis_plots",
]
__version__ = "0.1.0"
