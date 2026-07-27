"""Offline construction of a version-locked Radar-DINO reference catalog."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .catalog import ReferenceCatalog, _normalized_rows


def fit_reference_catalog(
    features: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
    *,
    pca_components: int = 50,
    min_cluster_size: int = 50,
    min_samples: int = 10,
    umap_neighbors: int = 30,
    umap_min_dist: float = 0.05,
    random_state: int = 42,
    include_tsne: bool = True,
) -> ReferenceCatalog:
    """Fit PCA, HDBSCAN, UMAP, and an optional display-only t-SNE."""

    try:
        import hdbscan
        import umap
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise ImportError(
            "Building a reference catalog requires the 'analysis' extra"
        ) from exc

    normalized = _normalized_rows(features)
    if len(normalized) < 3:
        raise ValueError("At least three reference features are required")
    component_count = min(
        int(pca_components),
        normalized.shape[1],
        normalized.shape[0] - 1,
    )
    if component_count <= 0:
        raise ValueError("pca_components must be positive")
    pca = PCA(
        n_components=component_count,
        svd_solver="full",
        random_state=random_state,
    )
    reduced = pca.fit_transform(normalized)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        prediction_data=True,
    ).fit(reduced)
    mapper = umap.UMAP(
        n_components=2,
        n_neighbors=min(umap_neighbors, len(reduced) - 1),
        min_dist=umap_min_dist,
        metric="euclidean",
        random_state=random_state,
    ).fit(reduced)

    reference_tsne = None
    if include_tsne:
        perplexity = min(30.0, max(1.0, (len(reduced) - 1) / 3.0))
        reference_tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        ).fit_transform(reduced)

    return ReferenceCatalog(
        normalized,
        metadata,
        pca=pca,
        clusterer=clusterer,
        umap=mapper,
        reference_tsne=reference_tsne,
    )


def save_reference_catalog(
    catalog: ReferenceCatalog,
    directory: str | Path,
) -> None:
    """Save fitted catalog data into an existing model artifact directory."""

    try:
        import joblib
    except ImportError as exc:
        raise ImportError(
            "Saving a fitted reference catalog requires the 'analysis' extra"
        ) from exc

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "reference_features.npy", catalog.features, allow_pickle=False)
    with (root / "reference_metadata.jsonl").open("w", encoding="utf-8") as stream:
        for row in catalog.metadata:
            stream.write(json.dumps(row, sort_keys=True))
            stream.write("\n")
    if catalog.pca is not None:
        joblib.dump(catalog.pca, root / "pca.joblib")
    if catalog.clusterer is not None:
        joblib.dump(catalog.clusterer, root / "hdbscan.joblib")
    if catalog.umap is not None:
        joblib.dump(catalog.umap, root / "umap.joblib")
    if catalog.reference_tsne is not None:
        np.save(root / "reference_tsne.npy", catalog.reference_tsne, allow_pickle=False)
