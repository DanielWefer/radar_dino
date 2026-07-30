"""Publication-ready PNG plots for Radar-DINO analysis results."""

from __future__ import annotations

from importlib import resources
from math import ceil
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


def chasespectral_colormap():
    """Return the bundled colorblind-friendly ChaseSpectral colormap."""

    _pyplot()
    from matplotlib.colors import LinearSegmentedColormap

    table = resources.files("radar_dino").joinpath(
        "data",
        "chase-spectral-rgb.txt",
    )
    with table.open("r", encoding="utf-8") as stream:
        rgb_values = np.loadtxt(stream)
    return LinearSegmentedColormap.from_list("ChaseSpectral", rgb_values)


def _radar_display_values(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(field, dtype=np.float32).copy()
    missing = ~np.isfinite(values) | (values < 0.0)
    values[missing] = np.nan
    return values, missing


def plot_attention_heads(
    result: RadarDINOResult,
    field: str,
    *,
    radar_field: np.ndarray,
    attention_cmap: str = "magma",
):
    """Plot a radar field followed by all of its attention heads.

    The released ``vit_small`` model has six heads, producing the requested
    4-by-2 layout. Other architectures use the same two-column style with
    enough rows for all heads.
    """

    plt = _pyplot()
    if result.attention is None:
        raise ValueError("Attention was not requested for this result")
    try:
        field_index = result.fields.index(field)
    except ValueError as exc:
        raise KeyError(f"Unknown radar field '{field}'") from exc

    attentions = np.asarray(result.attention[:, field_index])
    radar_field = np.asarray(radar_field)
    if attentions.ndim != 3:
        raise ValueError(
            "Expected field attention with shape heads x height x width, "
            f"got {attentions.shape}"
        )
    if radar_field.shape != attentions.shape[-2:]:
        raise ValueError(
            f"Radar field has shape {radar_field.shape}, "
            f"expected {attentions.shape[-2:]}"
        )

    finite_attention = attentions[np.isfinite(attentions)]
    if finite_attention.size == 0:
        raise ValueError("Attention maps contain no finite values to plot")
    attention_max = max(float(finite_attention.max()), 1.0e-12)

    num_heads = attentions.shape[0]
    rows = ceil((num_heads + 1) / 2)
    figure, axes = plt.subplots(
        rows,
        2,
        figsize=(12, 4.5 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()

    values, missing = _radar_display_values(radar_field)
    if field == "reflectivity":
        radar_cmap = chasespectral_colormap()
        radar_label = "normalized reflectivity"
    else:
        radar_cmap = plt.get_cmap("viridis")
        radar_label = "normalized value"
    radar_cmap = radar_cmap.copy()
    radar_cmap.set_bad(color="black")

    radar_image = flat_axes[0].imshow(
        values,
        cmap=radar_cmap,
        vmin=0.0,
        vmax=1.0,
        origin="upper",
        interpolation="none",
    )
    if missing.any():
        flat_axes[0].contour(
            missing.astype(float),
            levels=[0.5],
            colors="white",
            linewidths=0.4,
        )
    flat_axes[0].set_title(field.replace("_", " ").title())
    flat_axes[0].set_axis_off()
    figure.colorbar(
        radar_image,
        ax=flat_axes[0],
        fraction=0.046,
        pad=0.04,
        label=radar_label,
    )

    cmap = plt.get_cmap(attention_cmap).copy()
    cmap.set_bad(color="black")
    for head, axis in enumerate(flat_axes[1 : num_heads + 1]):
        image = axis.imshow(
            attentions[head],
            cmap=cmap,
            vmin=0.0,
            vmax=attention_max,
            origin="upper",
            interpolation="none",
        )
        axis.set_title(f"Attention head {head}")
        axis.set_axis_off()
        figure.colorbar(
            image,
            ax=axis,
            fraction=0.046,
            pad=0.04,
            label="attention",
        )

    for axis in flat_axes[num_heads + 1 :]:
        axis.set_axis_off()
    return figure, axes


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
            radar_field = None if radar_fields is None else radar_fields.get(field)
            if radar_field is None:
                figure, _ = plot_attention(result, field)
            else:
                figure, _ = plot_attention_heads(
                    result,
                    field,
                    radar_field=radar_field,
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
