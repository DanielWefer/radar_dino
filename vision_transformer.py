"""Backward-compatible imports for the packaged field-token transformer.

New code should import :mod:`radar_dino.vision_transformer`. This shim keeps
the historical top-level training scripts runnable from a source checkout.
"""

import sys
from pathlib import Path

_SOURCE_ROOT = str(Path(__file__).resolve().parent / "src")
if _SOURCE_ROOT not in sys.path:
    sys.path.insert(0, _SOURCE_ROOT)

from radar_dino.vision_transformer import *  # noqa: F401,F403
