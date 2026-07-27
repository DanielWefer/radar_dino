"""Radar-DINO feature and field-attention inference."""

from .artifact import export_checkpoint
from .catalog import ReferenceCatalog
from .config import RadarDINOConfig
from .model import RadarDINO
from .result import RadarDINOResult

__all__ = [
    "RadarDINO",
    "RadarDINOConfig",
    "RadarDINOResult",
    "ReferenceCatalog",
    "export_checkpoint",
]
__version__ = "0.1.0"
