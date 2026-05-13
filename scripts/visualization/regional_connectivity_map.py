#!/usr/bin/env python3
"""
Regional connectivity map (leader → follower) with anatomical orientation.

This script aggregates lead–lag latency pairs into region-level directed flow,
and places each region node at its spatial centroid computed from the electrode layout.

Inputs (default: --data-dir)
- lead_lag_pairwise.csv (channel_i, channel_j, latency_ms)
- lead_lag_bipolar_channel_labels.csv (bipolar_channel, label_a, label_b, ...)
- layout CSV with coordinates (default: BASE_DIR/bipolar_layout.csv)

Outputs (to --out-dir)
- region_centroids.csv
- region_flow_edges.csv
- region_latency_matrix_ms.csv (ordered regions)
- regional_connectivity_map.png
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import math

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable


DEFAULT_BASE_DIR = Path(
    "/vol/brains/bd3/pesaranlab/Archie_RecStim_vSUBNETS220_2nd/matlab/mfiles/Chen"
)
DEFAULT_DATA_DIR = DEFAULT_BASE_DIR / "session134" / "trial001_bipolar"


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

# Group labels into the 4 hubs you described (still keeping 9 nodes for plotting).
HUBS: dict[str, str] = {
    "r_medial_orbital_gyrus": "PFC Hub",
    "r_lateral_orbital_gyrus": "PFC Hub",
    "r_middle_frontal_gyrus": "PFC Hub",
    "r_superior_frontal_gyrus": "PFC Hub",
    "r_anterior_cingulate_gyrus": "Limbic/ACC",
    "r_precuneus": "Parietal Hub",
    "r_superior_parietal_lobule": "Parietal Hub",
    "r_supramarginal_gyrus": "Parietal Hub",
    "r_nucleus_accumbens": "Striatal",
}

DISPLAY_NAME: dict[str, str] = {
    "r_medial_orbital_gyrus": "mOFG",
    "r_lateral_orbital_gyrus": "lOFG",
    "r_middle_frontal_gyrus": "MFG",
    "r_superior_frontal_gyrus": "SFG",
    "r_anterior_cingulate_gyrus": "ACC",
    "r_precuneus": "Precuneus",
    "r_superior_parietal_lobule": "SPL",
    "r_supramarginal_gyrus": "SMG",
    "r_nucleus_accumbens": "NAcc",
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


def resolve_region_label(label_a: str, label_b: str, roi_set: set[str], strategy: str) -> str:
    la = (label_a or "unknown").strip()
    lb = (label_b or "unknown").strip()
    if strategy == "a":
        return la
    if strategy == "b":
        return lb
    # roi_prefer
    if la in roi_set:
        return la
    if lb in roi_set:
        return lb
    return la


def load_bipolar_to_region(mapping_csv: Path, roi_labels: list[str], label_strategy: str) -> dict[str, str]:
    roi_set = set(roi_labels)
    out: dict[str, str] = {}
    with open(mapping_csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ch = (row.get("bipolar_channel") or "").strip()
            if not ch:
                continue
            la = (row.get("label_a") or "unknown").strip()
            lb = (row.get("label_b") or "unknown").strip()
            region = resolve_region_label(la, lb, roi_set, label_strategy)
            if region in roi_set:
                out[ch] = region
    return out


def load_layout(layout_csv: Path, use_pixels: bool) -> dict[str, tuple[float, float]]:
    """
    Returns bipolar_ch -> (x, y)
    """
    ch_to_xy: dict[str, tuple[float, float]] = {}
    with open(layout_csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ch = (row.get("bipolar_ch") or "").strip()
            if not ch:
                continue
            xk = "Xpix" if use_pixels else "X"
            yk = "Ypix" if use_pixels else "Y"
            x = _finite_float(str(row.get(xk, "")).strip())
            y = _finite_float(str(row.get(yk, "")).strip())
            if x is None or y is None:
                continue
            ch_to_xy[ch] = (float(x), float(y))
    return ch_to_xy


def compute_region_centroids(
    roi_labels: list[str],
    bipolar_to_region: dict[str, str],
    ch_to_xy: dict[str, tuple[float, float]],
) -> tuple[dict[str, tuple[float, float]], dict[str, int]]:
    xs: dict[str, list[float]] = {r: [] for r in roi_labels}
    ys: dict[str, list[float]] = {r: [] for r in roi_labels}
    for ch, region in bipolar_to_region.items():
        if region not in xs:
            continue
        xy = ch_to_xy.get(ch)
        if xy is None:
            continue
        xs[region].append(float(xy[0]))
        ys[region].append(float(xy[1]))

    centroids: dict[str, tuple[float, float]] = {}
    counts: dict[str, int] = {}
    for r in roi_labels:
        counts[r] = len(xs[r])
        if len(xs[r]) == 0:
            continue
        centroids[r] = (float(np.mean(xs[r])), float(np.mean(ys[r])))
    return centroids, counts


def load_pairs_by_region(
    pairs_csv: Path, bipolar_to_region: dict[str, str]
) -> list[tuple[str, str, float]]:
    """
    Returns (region_i, region_j, latency_ms) for ROI-mapped channels only.
    """
    out: list[tuple[str, str, float]] = []
    with open(pairs_csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ci = (row.get("channel_i") or "").strip()
            cj = (row.get("channel_j") or "").strip()
            if not ci or not cj or ci == cj:
                continue
            ri = bipolar_to_region.get(ci)
            rj = bipolar_to_region.get(cj)
            if ri is None or rj is None:
                continue
            lat = _finite_float(str(row.get("latency_ms", "")).strip())
            if lat is None:
                continue
            out.append((ri, rj, float(lat)))
    return out


def compute_region_latency_matrix(regions: list[str], pairs: list[tuple[str, str, float]]) -> np.ndarray:
    idx = {r: i for i, r in enumerate(regions)}
    sums = np.zeros((len(regions), len(regions)), dtype=np.float64)
    counts = np.zeros((len(regions), len(regions)), dtype=np.int64)
    for ri, rj, lat in pairs:
        i = idx[ri]
        j = idx[rj]
        sums[i, j] += float(lat)
        counts[i, j] += 1
    M = np.full((len(regions), len(regions)), np.nan, dtype=np.float64)
    mask = counts > 0
    M[mask] = sums[mask] / counts[mask]
    np.fill_diagonal(M, 0.0)
    return M


def compute_pairwise_flow(M: np.ndarray, regions: list[str]) -> list[dict[str, object]]:
    """
    Collapse M into one directed edge per unordered pair using an antisymmetric flow:
      flow(A,B) = (M[A,B] - M[B,A]) / 2
    Positive => A leads B.
    """
    edges: list[dict[str, object]] = []
    n = len(regions)
    for i in range(n):
        for j in range(i + 1, n):
            a = float(M[i, j]) if math.isfinite(float(M[i, j])) else float("nan")
            b = float(M[j, i]) if math.isfinite(float(M[j, i])) else float("nan")
            if math.isnan(a) and math.isnan(b):
                continue
            # If one direction missing, fall back to the available mean.
            if math.isnan(a) and not math.isnan(b):
                flow = -b
            elif not math.isnan(a) and math.isnan(b):
                flow = a
            else:
                flow = (a - b) / 2.0

            if flow == 0 or not math.isfinite(flow):
                continue
            if flow > 0:
                src, dst, mag = regions[i], regions[j], flow
            else:
                src, dst, mag = regions[j], regions[i], -flow
            edges.append(
                {
                    "src_region": src,
                    "dst_region": dst,
                    "mean_latency_ms": float(mag),
                    "hub_src": HUBS.get(src, "Other"),
                    "hub_dst": HUBS.get(dst, "Other"),
                }
            )
    return edges


def save_centroids(out_dir: Path, regions: list[str], centroids: dict[str, tuple[float, float]], counts: dict[str, int]) -> Path:
    out_path = out_dir / "region_centroids.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["region", "display", "hub", "n_channels", "x", "y"])
        w.writeheader()
        for r in regions:
            xy = centroids.get(r)
            w.writerow(
                {
                    "region": r,
                    "display": DISPLAY_NAME.get(r, r),
                    "hub": HUBS.get(r, "Other"),
                    "n_channels": int(counts.get(r, 0)),
                    "x": "" if xy is None else f"{xy[0]:.6f}",
                    "y": "" if xy is None else f"{xy[1]:.6f}",
                }
            )
    return out_path


def save_edges(out_dir: Path, edges: list[dict[str, object]]) -> Path:
    out_path = out_dir / "region_flow_edges.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["src_region", "dst_region", "mean_latency_ms", "hub_src", "hub_dst"],
        )
        w.writeheader()
        for e in edges:
            w.writerow(e)
    return out_path


def save_region_matrix(out_dir: Path, regions: list[str], M: np.ndarray) -> Path:
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


def plot_connectivity(
    out_png: Path,
    regions: list[str],
    centroids: dict[str, tuple[float, float]],
    counts: dict[str, int],
    edges: list[dict[str, object]],
    anterior_up: bool,
    title: str,
    dpi: int = 300,
) -> None:
    # Filter out regions missing coordinates
    regions_plot = [r for r in regions if r in centroids]
    if len(regions_plot) < 2:
        raise ValueError("Not enough regions have centroids to plot.")

    # Compute magnitude scaling
    mags = [float(e["mean_latency_ms"]) for e in edges if math.isfinite(float(e["mean_latency_ms"]))]
    max_mag = max(mags) if mags else 1.0

    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw edges first (behind nodes)
    cmap = plt.get_cmap("Reds")
    norm = colors.Normalize(vmin=0.0, vmax=max_mag)

    for e in edges:
        src = str(e["src_region"])
        dst = str(e["dst_region"])
        mag = float(e["mean_latency_ms"])
        if src not in centroids or dst not in centroids:
            continue
        x1, y1 = centroids[src]
        x2, y2 = centroids[dst]

        lw = 0.8 + 4.0 * (mag / max_mag)
        col = cmap(norm(mag))

        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                color=col,
                linewidth=lw,
                alpha=0.85,
                shrinkA=16,
                shrinkB=16,
                mutation_scale=12 + 10 * (mag / max_mag),
            ),
            zorder=1,
        )

    # Draw nodes
    for r in regions_plot:
        x, y = centroids[r]
        n = int(counts.get(r, 0))
        size = 550 + 90 * min(n, 20)
        ax.scatter([x], [y], s=size, c="white", edgecolors="black", linewidths=1.2, zorder=3)
        label = DISPLAY_NAME.get(r, r)
        ax.text(x, y, label, ha="center", va="center", fontsize=10, fontweight="bold", zorder=4)

    # Make it look like a brain schematic
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

    if anterior_up:
        ax.invert_yaxis()

    # Colorbar for magnitude
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Mean lead–lag magnitude (ms)")

    plt.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Regional connectivity map with anatomical centroids")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--pairs-csv", type=Path, default=None, help="Default: data-dir/lead_lag_pairwise.csv")
    p.add_argument("--mapping-csv", type=Path, default=None, help="Default: data-dir/lead_lag_bipolar_channel_labels.csv")
    p.add_argument(
        "--layout-csv",
        type=Path,
        default=None,
        help="Electrode layout CSV with X/Y coords (default: BASE_DIR/bipolar_layout.csv)",
    )
    p.add_argument("--out-dir", type=Path, default=None, help="Default: data-dir/regional_connectivity_map")
    p.add_argument("--roi-labels", type=str, default=None, help="Comma-separated ROI labels (default: 9 labels)")
    p.add_argument(
        "--label-strategy",
        type=str,
        default="roi_prefer",
        choices=["roi_prefer", "a", "b"],
        help="How to assign a bipolar channel to a single region label.",
    )
    p.add_argument(
        "--abs-threshold-ms",
        type=float,
        default=5.0,
        help="Keep edges with |mean latency| > threshold (ms).",
    )
    p.add_argument(
        "--use-pixels",
        action="store_true",
        help="Use Xpix/Ypix instead of normalized X/Y for node placement.",
    )
    p.add_argument(
        "--anterior-up",
        action="store_true",
        help="Invert Y axis so anterior is at top (recommended for pixel coords).",
    )
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    pairs_csv = Path(args.pairs_csv) if args.pairs_csv is not None else data_dir / "lead_lag_pairwise.csv"
    mapping_csv = (
        Path(args.mapping_csv) if args.mapping_csv is not None else data_dir / "lead_lag_bipolar_channel_labels.csv"
    )
    layout_csv = Path(args.layout_csv) if args.layout_csv is not None else DEFAULT_BASE_DIR / "bipolar_layout.csv"
    out_dir = Path(args.out_dir) if args.out_dir is not None else data_dir / "regional_connectivity_map"
    out_dir.mkdir(parents=True, exist_ok=True)

    roi_labels = _parse_csv_list(args.roi_labels) if args.roi_labels else list(ROI_LABELS_DEFAULT)

    if not pairs_csv.exists():
        raise FileNotFoundError(f"Pairs CSV not found: {pairs_csv}")
    if not mapping_csv.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {mapping_csv}")
    if not layout_csv.exists():
        raise FileNotFoundError(f"Layout CSV not found: {layout_csv}")

    bipolar_to_region = load_bipolar_to_region(mapping_csv, roi_labels, args.label_strategy)
    if len(bipolar_to_region) == 0:
        raise ValueError("No bipolar channels mapped into the requested ROI labels.")

    ch_to_xy = load_layout(layout_csv, use_pixels=bool(args.use_pixels))
    centroids, counts = compute_region_centroids(roi_labels, bipolar_to_region, ch_to_xy)

    pairs = load_pairs_by_region(pairs_csv, bipolar_to_region)
    if len(pairs) == 0:
        raise ValueError("No pairs remained after ROI mapping.")

    regions = [r for r in roi_labels if r in centroids] + [r for r in roi_labels if r not in centroids]
    M = compute_region_latency_matrix(regions, pairs)

    edges = compute_pairwise_flow(M, regions)
    edges = [e for e in edges if abs(float(e["mean_latency_ms"])) > float(args.abs_threshold_ms)]

    cent_path = save_centroids(out_dir, regions, centroids, counts)
    mat_path = save_region_matrix(out_dir, regions, M)
    edge_path = save_edges(out_dir, edges)
    print(f"Saved: {cent_path}")
    print(f"Saved: {mat_path}")
    print(f"Saved: {edge_path}")
    print(f"Edges kept (|lat|>{args.abs_threshold_ms}ms): {len(edges)}")

    out_png = out_dir / "regional_connectivity_map.png"
    title = "Regional Connectivity Map (lead → follow)"
    plot_connectivity(
        out_png,
        regions=regions,
        centroids=centroids,
        counts=counts,
        edges=edges,
        anterior_up=bool(args.anterior_up),
        title=title,
        dpi=int(args.dpi),
    )
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()

