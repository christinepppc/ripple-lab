#!/usr/bin/env python3
"""
Leader–Follower visualization for ripple lead–lag pairs, aggregated by anatomy.

Inputs
- lead_lag_pairwise.csv: channel_i, channel_j, latency_ms (optionally extra columns)
- lead_lag_bipolar_channel_labels.csv: bipolar_channel,label_a,label_b,(...)

Outputs (to --out-dir)
- region_net_lead_score.csv
- region_latency_matrix_ms.csv
- leader_follower_net_lead_bar.png
- leader_follower_region_heatmap.png
- channel_to_region_mapping.csv
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import math
from typing import Iterable

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors


DEFAULT_DATA_DIR = Path(
    "/vol/brains/bd3/pesaranlab/Archie_RecStim_vSUBNETS220_2nd/matlab/mfiles/Chen/session134/trial001_bipolar"
)


ROI_LABELS_DEFAULT: list[str] = [
    "r_medial_orbital_gyrus",
    "r_lateral_orbital_gyrus",
    "r_middle_frontal_gyrus",
    "r_superior_frontal_gyrus",
    "r_anterior_cingulate_gyrus",
    "r_precuneus",
    "r_superior_parietal_lobule",
    "r_supramarginal_gyrus",
    "r_nucleus_accumbens",
]


HUBS: dict[str, str] = {
    # PFC Hub
    "r_medial_orbital_gyrus": "PFC Hub",
    "r_lateral_orbital_gyrus": "PFC Hub",
    "r_middle_frontal_gyrus": "PFC Hub",
    "r_superior_frontal_gyrus": "PFC Hub",
    # Limbic/ACC
    "r_anterior_cingulate_gyrus": "Limbic/ACC",
    # Parietal Hub
    "r_precuneus": "Parietal Hub",
    "r_superior_parietal_lobule": "Parietal Hub",
    "r_supramarginal_gyrus": "Parietal Hub",
    # Striatal
    "r_nucleus_accumbens": "Striatal",
}


def _parse_csv_list(value: str | None) -> list[str]:
    if value is None:
        return []
    out: list[str] = []
    for part in value.split(","):
        p = part.strip()
        if p:
            out.append(p)
    return out


def _finite_float(x: str) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def resolve_region_label(
    label_a: str,
    label_b: str,
    roi_set: set[str],
    strategy: str = "roi_prefer",
) -> str:
    """
    Map a bipolar channel to a single region label.

    strategy:
      - roi_prefer: if label_a in ROI -> label_a; else if label_b in ROI -> label_b; else label_a
      - a: always label_a
      - b: always label_b
    """
    la = (label_a or "unknown").strip()
    lb = (label_b or "unknown").strip()
    if strategy == "a":
        return la
    if strategy == "b":
        return lb
    if la in roi_set:
        return la
    if lb in roi_set:
        return lb
    return la


def load_channel_label_mapping(
    mapping_csv: Path,
    roi_labels: list[str],
    label_strategy: str = "roi_prefer",
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Returns:
      - channel_to_region: bipolar_channel -> ROI region label
      - channel_to_hub: bipolar_channel -> hub category
    Only returns channels that map into roi_labels.
    """
    roi_set = set(roi_labels)
    channel_to_region: dict[str, str] = {}
    channel_to_hub: dict[str, str] = {}

    with open(mapping_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ch = (row.get("bipolar_channel") or "").strip()
            if not ch:
                continue
            la = (row.get("label_a") or "unknown").strip()
            lb = (row.get("label_b") or "unknown").strip()
            region = resolve_region_label(la, lb, roi_set, strategy=label_strategy)
            if region not in roi_set:
                continue
            channel_to_region[ch] = region
            channel_to_hub[ch] = HUBS.get(region, "Other")

    return channel_to_region, channel_to_hub


def load_pairs(
    pairs_csv: Path,
    channel_to_region: dict[str, str],
) -> list[tuple[str, str, float, str, str]]:
    """
    Returns list of (channel_i, channel_j, latency_ms, region_i, region_j)
    filtered to channels present in channel_to_region.
    """
    pairs: list[tuple[str, str, float, str, str]] = []
    with open(pairs_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ci = (row.get("channel_i") or "").strip()
            cj = (row.get("channel_j") or "").strip()
            if not ci or not cj or ci == cj:
                continue
            if ci not in channel_to_region or cj not in channel_to_region:
                continue
            lat = _finite_float(str(row.get("latency_ms", "")).strip())
            if lat is None:
                continue
            ri = channel_to_region[ci]
            rj = channel_to_region[cj]
            pairs.append((ci, cj, float(lat), ri, rj))
    return pairs


def compute_channel_net_scores(
    pairs: Iterable[tuple[str, str, float, str, str]]
) -> dict[str, float]:
    """
    For each channel i, compute mean latency_ms over all (i, j) pairs.
    Positive mean => channel i tends to lead.
    """
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for ci, _cj, lat, _ri, _rj in pairs:
        sums[ci] = sums.get(ci, 0.0) + float(lat)
        counts[ci] = counts.get(ci, 0) + 1
    return {ch: (sums[ch] / counts[ch]) for ch in sums.keys() if counts.get(ch, 0) > 0}


def compute_region_net_scores(
    channel_net: dict[str, float],
    channel_to_region: dict[str, str],
    roi_labels: list[str],
) -> tuple[list[str], np.ndarray, dict[str, int]]:
    """
    Region net lead score = mean of per-channel net scores for channels in that region.
    Returns (regions_sorted, scores_sorted, n_channels_by_region).
    """
    by_region: dict[str, list[float]] = {r: [] for r in roi_labels}
    for ch, score in channel_net.items():
        r = channel_to_region.get(ch)
        if r in by_region:
            by_region[r].append(float(score))
    scores = {r: (float(np.mean(v)) if len(v) else float("nan")) for r, v in by_region.items()}
    n_channels = {r: len(by_region[r]) for r in by_region.keys()}
    regions_sorted = sorted(
        roi_labels,
        key=lambda r: (-np.inf if np.isnan(scores[r]) else scores[r]),
        reverse=True,
    )
    scores_sorted = np.array([scores[r] for r in regions_sorted], dtype=np.float64)
    return regions_sorted, scores_sorted, n_channels


def compute_region_latency_matrix(
    pairs: Iterable[tuple[str, str, float, str, str]],
    regions: list[str],
) -> np.ndarray:
    """
    M[i, j] = mean latency_ms for channel_i in region_i and channel_j in region_j.
    """
    idx = {r: k for k, r in enumerate(regions)}
    sums = np.zeros((len(regions), len(regions)), dtype=np.float64)
    counts = np.zeros((len(regions), len(regions)), dtype=np.int64)
    for _ci, _cj, lat, ri, rj in pairs:
        i = idx[ri]
        j = idx[rj]
        sums[i, j] += float(lat)
        counts[i, j] += 1
    M = np.full((len(regions), len(regions)), np.nan, dtype=np.float64)
    mask = counts > 0
    M[mask] = sums[mask] / counts[mask]
    np.fill_diagonal(M, 0.0)
    return M


def save_channel_mapping(
    out_dir: Path,
    channel_to_region: dict[str, str],
    channel_to_hub: dict[str, str],
) -> Path:
    out_path = out_dir / "channel_to_region_mapping.csv"
    rows = sorted(channel_to_region.keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bipolar_channel", "region", "hub"])
        w.writeheader()
        for ch in rows:
            w.writerow(
                {"bipolar_channel": ch, "region": channel_to_region[ch], "hub": channel_to_hub.get(ch, "Other")}
            )
    return out_path


def save_region_net_scores_csv(
    out_dir: Path,
    regions_sorted: list[str],
    scores_sorted: np.ndarray,
    n_channels: dict[str, int],
) -> Path:
    out_path = out_dir / "region_net_lead_score.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["region", "hub", "n_channels", "net_lead_score_ms"],
        )
        w.writeheader()
        for r, s in zip(regions_sorted, scores_sorted):
            w.writerow(
                {
                    "region": r,
                    "hub": HUBS.get(r, "Other"),
                    "n_channels": int(n_channels.get(r, 0)),
                    "net_lead_score_ms": "" if np.isnan(s) else f"{float(s):.4f}",
                }
            )
    return out_path


def save_region_matrix_csv(out_dir: Path, regions: list[str], M: np.ndarray) -> Path:
    out_path = out_dir / "region_latency_matrix_ms.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region_i\\region_j"] + regions)
        for i, ri in enumerate(regions):
            row = [ri]
            for j in range(len(regions)):
                v = M[i, j]
                row.append("" if np.isnan(v) else f"{float(v):.3f}")
            w.writerow(row)
    return out_path


def plot_net_lead_bar(
    out_path: Path,
    regions_sorted: list[str],
    scores_sorted: np.ndarray,
    n_channels: dict[str, int],
    figsize: tuple[float, float] = (10, 5.5),
    dpi: int = 300,
) -> None:
    y = np.arange(len(regions_sorted))
    colors_bar = ["firebrick" if (not np.isnan(s) and s > 0) else "royalblue" for s in scores_sorted]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(y, scores_sorted, color=colors_bar, edgecolor="white", linewidth=0.7)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{r}  (n={n_channels.get(r, 0)})" for r in regions_sorted],
        fontsize=9,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Net Lead Score (ms)  (positive = leads)")
    ax.set_title("Leader–Follower by Region (Net Lead Score)")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_region_heatmap(
    out_path: Path,
    regions: list[str],
    M: np.ndarray,
    figsize: tuple[float, float] = (8, 7),
    dpi: int = 300,
) -> None:
    vmax = float(np.nanmax(np.abs(M)))
    if not math.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    norm = colors.TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(M, cmap="RdBu_r", norm=norm, aspect="equal")
    ax.set_xticks(np.arange(len(regions)))
    ax.set_yticks(np.arange(len(regions)))
    ax.set_xticklabels(regions, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(regions, fontsize=8)
    ax.set_xlabel("Follower region (j)")
    ax.set_ylabel("Leader region (i)")
    ax.set_title("Mean Latency by Region (ms)")

    ax.set_xticks(np.arange(len(regions) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(regions) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="lightgray", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(len(regions)):
        for j in range(len(regions)):
            v = M[i, j]
            if np.isnan(v):
                continue
            txt = f"{float(v):.1f}"
            rgba = im.cmap(norm(v))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            c = "black" if lum > 0.6 else "white"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=c)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, label="Latency (ms)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Leader–Follower visualization aggregated by region")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Default directory for inputs/outputs")
    p.add_argument(
        "--pairs-csv",
        type=Path,
        default=None,
        help="Path to lead_lag_pairwise.csv (default: data-dir/lead_lag_pairwise.csv)",
    )
    p.add_argument(
        "--mapping-csv",
        type=Path,
        default=None,
        help="Path to lead_lag_bipolar_channel_labels.csv (default: data-dir/lead_lag_bipolar_channel_labels.csv)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: data-dir/leader_follower_regions)",
    )
    p.add_argument(
        "--roi-labels",
        type=str,
        default=None,
        help="Comma-separated ROI region labels (default: the 9 labels used in session 134 analysis).",
    )
    p.add_argument(
        "--label-strategy",
        type=str,
        default="roi_prefer",
        choices=["roi_prefer", "a", "b"],
        help="How to assign a bipolar channel to a single region label.",
    )
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--bar-figsize", type=float, nargs=2, default=[10, 5.5], metavar=("W", "H"))
    p.add_argument("--heatmap-figsize", type=float, nargs=2, default=[8, 7], metavar=("W", "H"))
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    pairs_csv = Path(args.pairs_csv) if args.pairs_csv is not None else data_dir / "lead_lag_pairwise.csv"
    mapping_csv = (
        Path(args.mapping_csv) if args.mapping_csv is not None else data_dir / "lead_lag_bipolar_channel_labels.csv"
    )
    out_dir = Path(args.out_dir) if args.out_dir is not None else data_dir / "leader_follower_regions"
    out_dir.mkdir(parents=True, exist_ok=True)

    roi_labels = _parse_csv_list(args.roi_labels) if args.roi_labels else list(ROI_LABELS_DEFAULT)

    if not pairs_csv.exists():
        raise FileNotFoundError(f"Pairs CSV not found: {pairs_csv}")
    if not mapping_csv.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {mapping_csv}")

    channel_to_region, channel_to_hub = load_channel_label_mapping(
        mapping_csv=mapping_csv,
        roi_labels=roi_labels,
        label_strategy=args.label_strategy,
    )
    if len(channel_to_region) < 2:
        raise ValueError(f"Only {len(channel_to_region)} channels mapped into ROI labels; cannot proceed.")

    mapping_out = save_channel_mapping(out_dir, channel_to_region, channel_to_hub)
    print(f"Saved mapping: {mapping_out}")

    pairs = load_pairs(pairs_csv, channel_to_region)
    if len(pairs) == 0:
        raise ValueError("No valid pairs found after filtering to ROI-mapped channels.")
    print(f"Loaded {len(pairs)} pairs after ROI filtering.")

    channel_net = compute_channel_net_scores(pairs)
    regions_sorted, scores_sorted, n_channels = compute_region_net_scores(channel_net, channel_to_region, roi_labels)

    scores_out = save_region_net_scores_csv(out_dir, regions_sorted, scores_sorted, n_channels)
    print(f"Saved region net lead scores: {scores_out}")

    M = compute_region_latency_matrix(pairs, regions_sorted)
    matrix_out = save_region_matrix_csv(out_dir, regions_sorted, M)
    print(f"Saved region latency matrix: {matrix_out}")

    bar_png = out_dir / "leader_follower_net_lead_bar.png"
    plot_net_lead_bar(
        bar_png,
        regions_sorted=regions_sorted,
        scores_sorted=scores_sorted,
        n_channels=n_channels,
        figsize=(float(args.bar_figsize[0]), float(args.bar_figsize[1])),
        dpi=int(args.dpi),
    )
    print(f"Saved: {bar_png}")

    heat_png = out_dir / "leader_follower_region_heatmap.png"
    plot_region_heatmap(
        heat_png,
        regions=regions_sorted,
        M=M,
        figsize=(float(args.heatmap_figsize[0]), float(args.heatmap_figsize[1])),
        dpi=int(args.dpi),
    )
    print(f"Saved: {heat_png}")


if __name__ == "__main__":
    main()

