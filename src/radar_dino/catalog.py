"""Reference embeddings, cluster assignment, projection, and scan similarity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _normalized_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got {array.shape}")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Reference features cannot contain zero-length vectors")
    return array / norms


def _normalized_query(feature: np.ndarray, dimension: int) -> np.ndarray:
    query = np.asarray(feature, dtype=np.float32)
    if query.ndim == 1:
        query = query[None, :]
    if query.ndim != 2 or query.shape[0] != 1 or query.shape[1] != dimension:
        raise ValueError(
            f"Expected one feature with dimension {dimension}, got {query.shape}"
        )
    norm = np.linalg.norm(query, axis=1, keepdims=True)
    if norm[0, 0] == 0:
        raise ValueError("Query feature cannot be a zero-length vector")
    return query / norm


class ReferenceCatalog:
    """A fitted reference population for downstream analysis of new scans."""

    def __init__(
        self,
        features: np.ndarray,
        metadata: Sequence[Mapping[str, Any]],
        *,
        pca=None,
        clusterer=None,
        umap=None,
        reference_tsne: np.ndarray | None = None,
    ) -> None:
        self.features = _normalized_rows(features)
        self.metadata = tuple(dict(row) for row in metadata)
        if len(self.metadata) != len(self.features):
            raise ValueError(
                f"Metadata has {len(self.metadata)} rows but features have "
                f"{len(self.features)} rows"
            )
        self.pca = pca
        self.clusterer = clusterer
        self.umap = umap
        self.reference_tsne = (
            None if reference_tsne is None else np.asarray(reference_tsne)
        )
        if (
            self.reference_tsne is not None
            and self.reference_tsne.shape != (len(self.features), 2)
        ):
            raise ValueError(
                "reference_tsne must have shape number_of_references x 2"
            )

    @classmethod
    def from_directory(cls, directory: str | Path) -> "ReferenceCatalog":
        root = Path(directory)
        features_path = root / "reference_features.npy"
        if not features_path.is_file():
            raise FileNotFoundError(f"Reference features do not exist: {features_path}")
        features = np.load(features_path, allow_pickle=False)

        parquet_path = root / "reference_metadata.parquet"
        jsonl_path = root / "reference_metadata.jsonl"
        if parquet_path.is_file():
            try:
                import pandas as pd
            except ImportError as exc:
                raise ImportError(
                    "Parquet reference metadata requires the 'analysis' extra"
                ) from exc
            metadata = pd.read_parquet(parquet_path).to_dict(orient="records")
        elif jsonl_path.is_file():
            with jsonl_path.open("r", encoding="utf-8") as stream:
                metadata = [json.loads(line) for line in stream if line.strip()]
        else:
            raise FileNotFoundError(
                f"Expected {parquet_path.name} or {jsonl_path.name} in {root}"
            )

        def load_joblib(name: str):
            path = root / name
            if not path.is_file():
                return None
            try:
                import joblib
            except ImportError as exc:
                raise ImportError(
                    "Fitted analysis models require the 'analysis' extra"
                ) from exc
            return joblib.load(path)

        tsne_path = root / "reference_tsne.npy"
        return cls(
            features,
            metadata,
            pca=load_joblib("pca.joblib"),
            clusterer=load_joblib("hdbscan.joblib"),
            umap=load_joblib("umap.joblib"),
            reference_tsne=(
                np.load(tsne_path, allow_pickle=False)
                if tsne_path.is_file()
                else None
            ),
        )

    @property
    def feature_dimension(self) -> int:
        return int(self.features.shape[1])

    def _query(self, feature: np.ndarray) -> np.ndarray:
        return _normalized_query(feature, self.feature_dimension)

    def _analysis_query(self, feature: np.ndarray) -> np.ndarray:
        query = self._query(feature)
        return self.pca.transform(query) if self.pca is not None else query

    def similar(
        self,
        feature: np.ndarray,
        *,
        k: int = 5,
        exclude_scan_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if k <= 0:
            raise ValueError("k must be positive")
        similarities = (self.features @ self._query(feature)[0]).astype(np.float64)
        candidate_indices = np.arange(len(self.features))
        if exclude_scan_id is not None:
            keep = np.array(
                [row.get("scan_id") != exclude_scan_id for row in self.metadata],
                dtype=bool,
            )
            candidate_indices = candidate_indices[keep]
        if candidate_indices.size == 0:
            return []
        candidate_scores = similarities[candidate_indices]
        count = min(k, candidate_indices.size)
        partial = np.argpartition(candidate_scores, -count)[-count:]
        ordered = partial[np.argsort(candidate_scores[partial])[::-1]]

        results = []
        for position in ordered:
            reference_index = int(candidate_indices[position])
            row = dict(self.metadata[reference_index])
            row["reference_index"] = reference_index
            row["similarity"] = float(similarities[reference_index])
            results.append(row)
        return results

    def project_umap(self, feature: np.ndarray) -> np.ndarray:
        if self.umap is None:
            raise RuntimeError("This reference catalog does not include a fitted UMAP model")
        return np.asarray(self.umap.transform(self._analysis_query(feature))[0])

    def predict_cluster(self, feature: np.ndarray) -> tuple[int, float]:
        if self.clusterer is None:
            raise RuntimeError(
                "This reference catalog does not include a fitted cluster model"
            )
        query = self._analysis_query(feature)
        try:
            import hdbscan
        except ImportError as exc:
            raise ImportError(
                "HDBSCAN prediction requires the 'analysis' extra"
            ) from exc
        labels, strengths = hdbscan.approximate_predict(self.clusterer, query)
        return int(labels[0]), float(strengths[0])
