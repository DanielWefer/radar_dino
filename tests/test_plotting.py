from __future__ import annotations

import numpy as np
import pytest

from radar_dino import RadarDINOResult, ReferenceCatalog
from radar_dino.plotting import save_analysis_plots


pytestmark = pytest.mark.unit


def test_save_analysis_plots_writes_attention_and_projection_pngs(tmp_path):
    pytest.importorskip("matplotlib")
    features = np.eye(3, dtype=np.float32)
    catalog = ReferenceCatalog(
        features,
        [{"scan_id": f"scan-{index}"} for index in range(3)],
        reference_umap=np.array([[0, 0], [1, 1], [2, 0]], dtype=np.float32),
        reference_tsne=np.array([[0, 2], [1, 3], [2, 2]], dtype=np.float32),
        reference_clusters=np.array([-1, 0, 1]),
    )
    result = RadarDINOResult(
        path=tmp_path / "scan.nc",
        fields=("reflectivity", "spectrum_width"),
        feature=features[1],
        attention=np.ones((2, 2, 4, 4), dtype=np.float32),
        umap=np.array([1.1, 0.9], dtype=np.float32),
        tsne=np.array([1.2, 2.8], dtype=np.float32),
        cluster=0,
    )

    saved = save_analysis_plots(
        result,
        catalog,
        tmp_path,
        radar_fields={
            "reflectivity": np.zeros((4, 4), dtype=np.float32),
            "spectrum_width": np.ones((4, 4), dtype=np.float32),
        },
        dpi=80,
    )

    assert set(saved) == {
        "attention_reflectivity",
        "attention_spectrum_width",
        "umap",
        "tsne",
    }
    for path in saved.values():
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
