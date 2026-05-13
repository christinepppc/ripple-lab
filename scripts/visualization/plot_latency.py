#!/usr/bin/env python3
"""
Plot 78×78 lead-lag latency matrix as a publication-quality clustermap.

Loads lead_lag_latency_ms.npy and channel names from .mat (or b001..b078),
plots with sns.clustermap (diverging RdBu_r, centered at 0 ms), and saves
ripple_propagation_map.png at 300 DPI.
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import scipy.io as sio
from scipy.cluster import hierarchy

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

DEFAULT_DATA_DIR = Path(
    "/vol/brains/bd3/pesaranlab/Archie_RecStim_vSUBNETS220_2nd/matlab/mfiles/Chen/session134/trial001_bipolar"
)


def load_latency_and_labels(data_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Load latency matrix from .npy and channel names from .mat or generate b001..b078."""
    npy_path = data_dir / "lead_lag_latency_ms.npy"
    if not npy_path.exists():
        raise FileNotFoundError(f"Latency matrix not found: {npy_path}")
    latency = np.load(npy_path)

    mat_path = data_dir / "lead_lag_latency_ms.mat"
    channel_names = None
    if mat_path.exists():
        try:
            data = sio.loadmat(str(mat_path))
            ch = data.get("channels")
            if ch is not None:
                # MATLAB may return (1, n) or (n,) array of arrays
                ch = np.squeeze(ch)
                if ch.ndim == 0:
                    channel_names = [str(ch)]
                else:
                    channel_names = [str(np.squeeze(c)) for c in ch.flat]
        except Exception:
            pass
    if channel_names is None or len(channel_names) != latency.shape[0]:
        n_ch = latency.shape[0]
        channel_names = [f"b{i:03d}" for i in range(1, n_ch + 1)]
    return latency, channel_names


def _plot_clustermap_matplotlib(
    lat_plot: np.ndarray,
    channel_names: list[str],
    out_path: Path,
    figsize: tuple[float, float],
    dpi: int,
    vmin: float,
    vmax: float,
) -> None:
    """Matplotlib + scipy hierarchy fallback when seaborn is not available."""
    n = lat_plot.shape[0]
    # Linkage for rows and cols (average method, Euclidean on the latency rows/cols)
    row_link = hierarchy.linkage(lat_plot, method="average", metric="euclidean")
    col_link = hierarchy.linkage(lat_plot.T, method="average", metric="euclidean")
    row_dend = hierarchy.dendrogram(row_link, no_plot=True)
    col_dend = hierarchy.dendrogram(col_link, no_plot=True)
    row_idx = np.array(row_dend["leaves"])
    col_idx = np.array(col_dend["leaves"])
    lat_reorder = lat_plot[np.ix_(row_idx, col_idx)]
    labels_row = [channel_names[i] for i in row_idx]
    labels_col = [channel_names[j] for j in col_idx]

    fig = plt.figure(figsize=figsize)
    # Dendrogram top
    ax_dend_col = fig.add_axes([0.15, 0.82, 0.7, 0.12])
    hierarchy.dendrogram(col_link, ax=ax_dend_col, color_threshold=0, above_threshold_color="k")
    ax_dend_col.axis("off")
    # Dendrogram left
    ax_dend_row = fig.add_axes([0.02, 0.15, 0.12, 0.65])
    hierarchy.dendrogram(row_link, ax=ax_dend_row, color_threshold=0, above_threshold_color="k", orientation="left")
    ax_dend_row.axis("off")
    # Heatmap
    ax_heat = fig.add_axes([0.15, 0.15, 0.7, 0.65])
    im = ax_heat.imshow(lat_reorder, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax_heat.set_xticks(np.arange(n))
    ax_heat.set_yticks(np.arange(n))
    ax_heat.set_xticklabels(labels_col, rotation=90, ha="right", fontsize=4)
    ax_heat.set_yticklabels(labels_row, fontsize=4)
    ax_heat.set_xlabel("Channel $j$")
    ax_heat.set_ylabel("Channel $i$")
    # Subtle grid
    ax_heat.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax_heat.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax_heat.grid(which="minor", color="lightgray", linewidth=0.2, linestyle="-")
    ax_heat.tick_params(which="minor", size=0)
    # Colorbar
    cax = fig.add_axes([0.88, 0.15, 0.02, 0.65])
    cbar = fig.colorbar(im, cax=cax, label="Latency (ms)")
    fig.suptitle("Ripple propagation (lead–lag)", y=0.98, fontsize=12)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap_ordered(
    latency: np.ndarray,
    channel_names: list[str],
    out_path: Path,
    figsize: tuple[float, float],
    dpi: int,
    vmax_ms: float | None = None,
) -> None:
    """
    Simple heatmap that keeps the original channel order (no clustering).
    """
    n = latency.shape[0]
    if vmax_ms is None:
        vmax_ms = float(np.nanmax(np.abs(latency)))
    if vmax_ms <= 0:
        vmax_ms = 50.0
    vmin, vmax = -vmax_ms, vmax_ms

    lat_plot = latency.copy()
    np.fill_diagonal(lat_plot, 0.0)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(lat_plot, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(channel_names, rotation=90, ha="right", fontsize=4)
    ax.set_yticklabels(channel_names, fontsize=4)
    ax.set_xlabel("Channel $j$")
    ax.set_ylabel("Channel $i$")

    # Subtle grid
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="lightgray", linewidth=0.2, linestyle="-")
    ax.tick_params(which="minor", size=0)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Latency (ms)")
    fig.suptitle("Ripple propagation (lead–lag, ordered channels)", y=0.98, fontsize=12)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_latency_clustermap(
    latency: np.ndarray,
    channel_names: list[str],
    out_path: Path,
    figsize: tuple[float, float] = (12, 10),
    dpi: int = 300,
    vmax_ms: float | None = None,
) -> None:
    """
    Plot latency matrix as clustermap (diverging RdBu_r, centered at 0 ms).
    Uses seaborn.clustermap if available, else matplotlib + scipy hierarchy.
    Diagonal is 0 ms; subtle gridlines separate channels.
    """
    n = latency.shape[0]
    if vmax_ms is None:
        vmax_ms = float(np.nanmax(np.abs(latency)))
    if vmax_ms <= 0:
        vmax_ms = 50.0
    vmin, vmax = -vmax_ms, vmax_ms

    lat_plot = latency.copy()
    np.fill_diagonal(lat_plot, 0.0)

    if HAS_SEABORN:
        import pandas as pd
        df = pd.DataFrame(lat_plot, index=channel_names, columns=channel_names)
        g = sns.clustermap(
            df,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            center=0,
            figsize=figsize,
            linewidths=0.2,
            linecolor="lightgray",
            dendrogram_ratio=(0.12, 0.12),
            cbar_pos=(0.02, 0.82, 0.03, 0.15),
            cbar_kws={"label": "Latency (ms)", "shrink": 0.6},
            xticklabels=True,
            yticklabels=True,
            method="average",
        )
        g.ax_heatmap.set_xlabel("Channel $j$")
        g.ax_heatmap.set_ylabel("Channel $i$")
        g.fig.suptitle("Ripple propagation (lead–lag)", y=1.02, fontsize=12)
        plt.tight_layout()
        g.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        _plot_clustermap_matplotlib(lat_plot, channel_names, out_path, figsize, dpi, vmin, vmax)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot lead-lag latency matrix as publication-quality clustermap"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing lead_lag_latency_ms.npy and .mat",
    )
    parser.add_argument(
        "--keep-order",
        action="store_true",
        help="Keep original channel order (no clustering) in the heatmap",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: data-dir/ripple_propagation_map.png)",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[12, 10],
        metavar=("W", "H"),
        help="Figure size (default: 12 10)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI")
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Symmetric color limit ±vmax ms (default: max abs value)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.output is not None:
        out_path = Path(args.output)
    else:
        # Different default name when keeping order
        name = "ripple_propagation_map_ordered.png" if args.keep_order else "ripple_propagation_map.png"
        out_path = data_dir / name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    latency, channel_names = load_latency_and_labels(data_dir)
    print(f"Loaded {latency.shape[0]}×{latency.shape[1]} matrix, {len(channel_names)} labels")

    if args.keep_order:
        _plot_heatmap_ordered(
            latency,
            channel_names,
            out_path,
            figsize=tuple(args.figsize),
            dpi=args.dpi,
            vmax_ms=args.vmax,
        )
        print(f"Saved (ordered heatmap): {out_path}")
    else:
        plot_latency_clustermap(
            latency,
            channel_names,
            out_path,
            figsize=tuple(args.figsize),
            dpi=args.dpi,
            vmax_ms=args.vmax,
        )


if __name__ == "__main__":
    main()
