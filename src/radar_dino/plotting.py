"""Publication-ready PNG plots for Radar-DINO analysis results."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from .catalog import ReferenceCatalog
from .result import RadarDINOResult


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "PNG plotting requires the 'plot' extra: "
            "python -m pip install 'radar-dino[plot]'"
        ) from exc
    return plt


def plot_attention(
    result: RadarDINOResult,
    field: str,
    *,
    head: int | str = "mean",
    radar_field: np.ndarray | None = None,
    cmap: str = "magma",
):
    """Plot one field's attention, optionally beside its normalized radar field."""

    plt = _pyplot()
    attention = result.attention_for(field, head=head)
    if radar_field is None:
        figure, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
        axes = np.asarray([axis], dtype=object)
    else:
        radar_field = np.asarray(radar_field)
        if radar_field.shape != attention.shape:
            raise ValueError(
                f"Radar field has shape {radar_field.shape}, expected {attention.shape}"
            )
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(11, 5),
            constrained_layout=True,
        )
        radar_image = axes[0].imshow(
            radar_field,
            origin="lower",
            interpolation="none",
            cmap="viridis",
        )
        axes[0].set_title(field.replace("_", " ").title())
        figure.colorbar(radar_image, ax=axes[0], shrink=0.82)

    attention_axis = axes[-1]
    attention_image = attention_axis.imshow(
        attention,
        origin="lower",
        interpolation="none",
        cmap=cmap,
    )
    head_label = "Mean-head" if head == "mean" else f"Head {head}"
    attention_axis.set_title(f"{head_label} attention: {field}")
    figure.colorbar(attention_image, ax=attention_axis, shrink=0.82)
    for axis in axes:
        axis.set_xlabel("x grid cell")
        axis.set_ylabel("y grid cell")
    return figure, axes


def plot_projection(
    catalog: ReferenceCatalog,
    result: RadarDINOResult,
    *,
    method: str,
):
    """Plot the fixed reference population and highlight one query scan."""

    plt = _pyplot()
    method_name = method.strip().lower()
    if method_name == "umap":
        references = catalog.reference_umap
        query = result.umap
        title = "Radar-DINO UMAP"
    elif method_name in {"tsne", "t-sne"}:
        references = catalog.reference_tsne
        query = result.tsne
        title = "Radar-DINO t-SNE (query position interpolated)"
        method_name = "tsne"
    else:
        raise ValueError("method must be 'umap' or 'tsne'")
    if references is None:
        raise RuntimeError(f"The reference catalog has no {method_name} coordinates")
    if query is None:
        raise RuntimeError(f"The result has no {method_name} coordinate")

    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    labels = catalog.reference_clusters
    if labels is None:
        axis.scatter(
            references[:, 0],
            references[:, 1],
            s=7,
            alpha=0.35,
            color="0.45",
            linewidths=0,
            label="Reference scans",
        )
    else:
        labels = np.asarray(labels)
        for label in sorted(np.unique(labels)):
            selected = labels == label
            if label == -1:
                color = "0.72"
                legend_label = "Noise"
                zorder = 1
            else:
                color = plt.get_cmap("tab10")(int(label) % 10)
                legend_label = f"Cluster {int(label)}"
                zorder = 2
            axis.scatter(
                references[selected, 0],
                references[selected, 1],
                s=7,
                alpha=0.42,
                color=color,
                linewidths=0,
                label=legend_label,
                zorder=zorder,
            )
    axis.scatter(
        query[0],
        query[1],
        marker="*",
        s=260,
        color="red",
        edgecolor="black",
        linewidth=0.8,
        label="Input scan",
        zorder=10,
    )
    axis.set_title(title)
    axis.set_xlabel(f"{method_name.upper()} 1")
    axis.set_ylabel(f"{method_name.upper()} 2")
    axis.legend(loc="best", frameon=True, markerscale=1.4)
    return figure, axis


def save_analysis_plots(
    result: RadarDINOResult,
    catalog: ReferenceCatalog | None,
    output_directory: str | Path,
    *,
    radar_fields: Mapping[str, np.ndarray] | None = None,
    dpi: int = 200,
) -> dict[str, Path]:
    """Save available attention and projection figures as PNG files."""

    if dpi <= 0:
        raise ValueError("dpi must be positive")
    plt = _pyplot()
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    if result.attention is not None:
        for field in result.fields:
            figure, _ = plot_attention(
                result,
                field,
                radar_field=(
                    None if radar_fields is None else radar_fields.get(field)
                ),
            )
            path = output / f"attention_{field}.png"
            figure.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(figure)
            saved[f"attention_{field}"] = path

    if catalog is not None and result.umap is not None:
        figure, _ = plot_projection(catalog, result, method="umap")
        path = output / "umap.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        saved["umap"] = path

    if catalog is not None and result.tsne is not None:
        figure, _ = plot_projection(catalog, result, method="tsne")
        path = output / "tsne.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        saved["tsne"] = path

    return saved
