#!/usr/bin/env python3
"""
Lead-Lag analysis on ripple events across bipolar channels.

Processes Session 134 Trial 001_bipolar: builds binary ripple series per channel,
bins at 10 ms, computes pairwise cross-correlation with FFT, and outputs a
78×78 latency matrix (ms) with optional significance masking.
"""

from pathlib import Path
import argparse
import csv
import re
import numpy as np
import scipy.io as sio
from scipy import signal
from scipy import stats

# Default paths
DEFAULT_ROOT = Path(
    "/vol/brains/bd3/pesaranlab/Archie_RecStim_vSUBNETS220_2nd/matlab/mfiles/Chen"
)
SESSION = 134
TRIAL = 1
BIN_MS = 10
LAG_WINDOW_BINS = 5   # ±50 ms at 10 ms bins
LAG_MS = BIN_MS * LAG_WINDOW_BINS  # 50 ms

# Session 134: default regions of interest (ROIs) for lead-lag analysis.
# These are labels found in session134/channel_labels.csv.
DEFAULT_ROIS_SESSION134: list[str] = [
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


def load_bipolar_layout_coords(
    layout_csv: Path, channels: list[str], use_pixels: bool = False
) -> dict[str, tuple[float, float]]:
    """
    Load bipolar channel coordinates from bipolar_layout.csv.

    Expected columns include:
      - bipolar_ch
      - X, Y (normalized) and/or Xpix, Ypix (pixel coordinates)
    """
    if not layout_csv.exists():
        raise FileNotFoundError(f"Layout CSV not found: {layout_csv}")

    x_key = "Xpix" if use_pixels else "X"
    y_key = "Ypix" if use_pixels else "Y"
    ch_set = set(channels)
    out: dict[str, tuple[float, float]] = {}

    with open(layout_csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ch = (row.get("bipolar_ch") or "").strip()
            if not ch or ch not in ch_set:
                continue
            try:
                x = float(row.get(x_key))  # type: ignore[arg-type]
                y = float(row.get(y_key))  # type: ignore[arg-type]
            except Exception:
                continue
            if np.isfinite(x) and np.isfinite(y):
                out[ch] = (x, y)
    return out


def build_bipolar_to_roi_label(
    bipolar_dirs: list[Path],
    channel_labels: dict[int, str],
    roi_labels: list[str],
    label_strategy: str = "roi_prefer",
) -> dict[str, str]:
    """
    Build mapping bipolar channel (b###) -> ROI label in roi_labels.

    label_strategy:
      - roi_prefer: if label_a in ROI -> label_a, else if label_b in ROI -> label_b, else label_a
      - a: always label_a
      - b: always label_b

    Uses `pair_info.txt` + `channel_labels.csv` mapping (original channel -> label).
    """
    roi_set = set(roi_labels)
    out: dict[str, str] = {}
    for d in bipolar_dirs:
        bch = d.name
        ch_a, ch_b = parse_pair_info(d / "pair_info.txt")
        la = channel_labels.get(ch_a, "unknown")
        lb = channel_labels.get(ch_b, "unknown")

        if label_strategy == "a":
            region = la
        elif label_strategy == "b":
            region = lb
        else:
            if la in roi_set:
                region = la
            elif lb in roi_set:
                region = lb
            else:
                region = la

        if region in roi_set:
            out[bch] = region
    return out


def compute_distance_flow_correlation(
    latency_ms: np.ndarray,
    channel_names: list[str],
    coords: dict[str, tuple[float, float]],
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Compute Spearman correlation between spatial distance and lead–lag "flow magnitude".

    We use an antisymmetric per-pair flow:
      flow(i,j) = (latency[i,j] - latency[j,i]) / 2
    and correlate distance(i,j) with |flow(i,j)| over i<j.

    Returns (rho, p_naive_two_sided, distances, flow_magnitudes).
    """
    n = len(channel_names)
    if latency_ms.shape[0] != n or latency_ms.shape[1] != n:
        raise ValueError("latency_ms shape does not match channel_names length")

    # Keep only channels with coordinates
    keep_idx = [i for i, ch in enumerate(channel_names) if ch in coords]
    if len(keep_idx) < 3:
        raise ValueError(f"Need at least 3 channels with coordinates; got {len(keep_idx)}")

    idx_map = {old: new for new, old in enumerate(keep_idx)}
    kept_names = [channel_names[i] for i in keep_idx]

    lat = latency_ms[np.ix_(keep_idx, keep_idx)]
    n2 = lat.shape[0]
    I, J = np.triu_indices(n2, 1)

    # Coordinates in kept order
    xy = np.array([coords[ch] for ch in kept_names], dtype=np.float64)
    d = xy[I] - xy[J]
    distances = np.sqrt(np.sum(d * d, axis=1))

    flow = (lat[I, J] - lat[J, I]) / 2.0
    flow_mag = np.abs(flow)

    finite = np.isfinite(distances) & np.isfinite(flow_mag)
    distances = distances[finite]
    flow_mag = flow_mag[finite]

    if len(distances) < 5:
        raise ValueError(f"Too few valid pairs for correlation: {len(distances)}")

    rho, p = stats.spearmanr(distances, flow_mag)
    return float(rho), float(p), distances, flow_mag


def coordinate_shuffle_null_for_distance_flow(
    latency_ms: np.ndarray,
    channel_names: list[str],
    coords: dict[str, tuple[float, float]],
    n_perm: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Geometry-control null: shuffle coordinates across channels and recompute
    Spearman rho(distance, |flow|). Latency matrix is unchanged.
    """
    rng = np.random.default_rng(seed)
    n = len(channel_names)

    keep_idx = [i for i, ch in enumerate(channel_names) if ch in coords]
    if len(keep_idx) < 3:
        raise ValueError(f"Need at least 3 channels with coordinates; got {len(keep_idx)}")

    kept_names = [channel_names[i] for i in keep_idx]
    lat = latency_ms[np.ix_(keep_idx, keep_idx)]
    n2 = lat.shape[0]
    I, J = np.triu_indices(n2, 1)

    xy = np.array([coords[ch] for ch in kept_names], dtype=np.float64)
    flow = (lat[I, J] - lat[J, I]) / 2.0
    flow_mag = np.abs(flow)

    finite_flow = np.isfinite(flow_mag)
    if np.sum(finite_flow) < 5:
        raise ValueError("Too few finite flow pairs for coordinate-shuffle null")

    I2 = I[finite_flow]
    J2 = J[finite_flow]
    flow_mag2 = flow_mag[finite_flow]

    null_rhos = np.zeros(int(n_perm), dtype=np.float64)
    for k in range(int(n_perm)):
        perm = rng.permutation(n2)
        xy_p = xy[perm]
        d = xy_p[I2] - xy_p[J2]
        distances = np.sqrt(np.sum(d * d, axis=1))
        rho, _p = stats.spearmanr(distances, flow_mag2)
        null_rhos[k] = float(rho)
    return null_rhos


def coordinate_shuffle_within_groups_null_for_distance_flow(
    latency_ms: np.ndarray,
    channel_names: list[str],
    coords: dict[str, tuple[float, float]],
    channel_to_group: dict[str, str],
    n_perm: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Geometry-control null (within-group): shuffle coordinates among channels *within each group*
    (e.g., within each ROI), and recompute Spearman rho(distance, |flow|).

    Latency matrix is unchanged; only the mapping from channel identity -> (x,y) is randomized
    within groups.
    """
    rng = np.random.default_rng(seed)

    keep_idx = [
        i
        for i, ch in enumerate(channel_names)
        if (ch in coords) and (ch in channel_to_group)
    ]
    if len(keep_idx) < 3:
        raise ValueError(f"Need at least 3 channels with coords+group; got {len(keep_idx)}")

    kept_names = [channel_names[i] for i in keep_idx]
    lat = latency_ms[np.ix_(keep_idx, keep_idx)]
    n2 = lat.shape[0]
    I, J = np.triu_indices(n2, 1)

    xy = np.array([coords[ch] for ch in kept_names], dtype=np.float64)
    flow = (lat[I, J] - lat[J, I]) / 2.0
    flow_mag = np.abs(flow)

    finite_flow = np.isfinite(flow_mag)
    if np.sum(finite_flow) < 5:
        raise ValueError("Too few finite flow pairs for within-group coordinate-shuffle null")

    I2 = I[finite_flow]
    J2 = J[finite_flow]
    flow_mag2 = flow_mag[finite_flow]

    # Build index lists per group in kept order
    groups: dict[str, list[int]] = {}
    for idx, ch in enumerate(kept_names):
        g = channel_to_group[ch]
        groups.setdefault(g, []).append(idx)

    null_rhos = np.zeros(int(n_perm), dtype=np.float64)
    for k in range(int(n_perm)):
        xy_p = xy.copy()
        for g, idxs in groups.items():
            if len(idxs) < 2:
                continue
            perm = rng.permutation(len(idxs))
            xy_p[np.array(idxs, dtype=int)] = xy_p[np.array(idxs, dtype=int)][perm]
        d = xy_p[I2] - xy_p[J2]
        distances = np.sqrt(np.sum(d * d, axis=1))
        rho, _p = stats.spearmanr(distances, flow_mag2)
        null_rhos[k] = float(rho)
    return null_rhos


def _parse_csv_list(value: str | None) -> list[str]:
    if value is None:
        return []
    items = []
    for part in value.split(","):
        p = part.strip()
        if p:
            items.append(p)
    return items


def load_channel_labels_csv(path: Path) -> dict[int, str]:
    """
    Load session-level channel labels mapping original channel -> label.
    Expected CSV format: channel,label (as in session134/channel_labels.csv)
    """
    labels: dict[int, str] = {}
    if not path.exists():
        return labels
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        # tolerate alternate header capitalization
        for row in reader:
            if not row:
                continue
            ch_raw = row.get("channel") or row.get("Channel") or row.get("ch") or row.get("Ch")
            label = row.get("label") or row.get("Label") or row.get("region") or row.get("Region")
            if ch_raw is None:
                continue
            try:
                ch = int(str(ch_raw).strip())
            except ValueError:
                continue
            labels[ch] = (str(label).strip() if label is not None else "unknown")
    return labels


def parse_pair_info(pair_info_path: Path) -> tuple[int, int]:
    """
    Parse bipolar channel `pair_info.txt` to extract the original channel pair.
    Line format: "Original channels: 1 - 2"
    """
    text = pair_info_path.read_text(errors="ignore")
    m = re.search(r"Original channels:\s*(\d+)\s*-\s*(\d+)", text)
    if not m:
        raise ValueError(f"Could not parse original channel pair from {pair_info_path}")
    return int(m.group(1)), int(m.group(2))


def _label_matches(
    label: str, patterns: list[str], match_mode: str = "substring"
) -> bool:
    """
    match_mode:
      - exact: case-insensitive equality
      - substring: case-insensitive substring containment
    """
    if not patterns:
        return False
    lab = (label or "").strip().lower()
    for p in patterns:
        pp = p.strip().lower()
        if not pp:
            continue
        if match_mode == "exact":
            if lab == pp:
                return True
        else:
            if pp in lab:
                return True
    return False


def filter_bipolar_dirs(
    bipolar_dirs: list[Path],
    channel_labels: dict[int, str],
    include_regions: list[str],
    exclude_regions: list[str],
    region_match: str,
    region_match_mode: str,
    include_channels: list[str],
    exclude_channels: list[str],
    out_dir: Path | None = None,
) -> tuple[list[Path], list[dict[str, str]]]:
    """
    Filter bipolar channel directories by regions (using channel_labels + pair_info.txt)
    and/or explicit channel include/exclude lists (b001,b002,...).

    Returns (filtered_dirs, metadata_rows). metadata_rows is written to CSV if out_dir is set.
    """
    include_set = {c.strip().lower() for c in include_channels if c.strip()}
    exclude_set = {c.strip().lower() for c in exclude_channels if c.strip()}

    rows: list[dict[str, str]] = []
    kept: list[Path] = []

    for d in bipolar_dirs:
        bipolar = d.name
        bipolar_l = bipolar.lower()

        ch_a, ch_b = parse_pair_info(d / "pair_info.txt")
        lab_a = channel_labels.get(ch_a, "unknown")
        lab_b = channel_labels.get(ch_b, "unknown")

        # explicit channel include/exclude
        explicitly_included = (not include_set) or (bipolar_l in include_set)
        explicitly_excluded = bipolar_l in exclude_set

        def matches_include(label: str) -> bool:
            return _label_matches(label, include_regions, match_mode=region_match_mode)

        def matches_exclude(label: str) -> bool:
            return _label_matches(label, exclude_regions, match_mode=region_match_mode)

        if include_regions:
            if region_match == "primary":
                region_include_ok = matches_include(lab_a)
            elif region_match == "both":
                region_include_ok = matches_include(lab_a) and matches_include(lab_b)
            else:  # either
                region_include_ok = matches_include(lab_a) or matches_include(lab_b)
        else:
            region_include_ok = True

        if exclude_regions:
            if region_match == "primary":
                region_excluded = matches_exclude(lab_a)
            elif region_match == "both":
                region_excluded = matches_exclude(lab_a) and matches_exclude(lab_b)
            else:  # either
                region_excluded = matches_exclude(lab_a) or matches_exclude(lab_b)
        else:
            region_excluded = False

        keep = explicitly_included and (not explicitly_excluded) and region_include_ok and (not region_excluded)

        rows.append(
            {
                "bipolar_channel": bipolar,
                "orig_ch_a": str(ch_a),
                "orig_ch_b": str(ch_b),
                "label_a": lab_a,
                "label_b": lab_b,
                "kept": "1" if keep else "0",
            }
        )
        if keep:
            kept.append(d)

    if out_dir is not None:
        out_path = Path(out_dir) / "lead_lag_bipolar_channel_labels.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["bipolar_channel", "orig_ch_a", "orig_ch_b", "label_a", "label_b", "kept"],
            )
            writer.writeheader()
            writer.writerows(rows)

    return kept, rows


def get_bipolar_dirs(root: Path, session: int, trial: int) -> list[Path]:
    """Return sorted list of bipolar channel directories b001, b002, ..."""
    trial_dir = root / f"session{session:03d}" / f"trial{trial:03d}_bipolar"
    if not trial_dir.exists():
        raise FileNotFoundError(f"Trial directory not found: {trial_dir}")
    dirs = sorted(trial_dir.glob("b[0-9][0-9][0-9]"))
    return dirs


def get_lfp_length_and_fs(bipolar_dir: Path) -> tuple[int, float]:
    """Load LFP from lfp_b0xx.mat and return length and sampling rate."""
    lfp_file = bipolar_dir / f"lfp_{bipolar_dir.name}.mat"
    if not lfp_file.exists():
        raise FileNotFoundError(f"LFP file not found: {lfp_file}")
    data = sio.loadmat(str(lfp_file))
    lfp = data.get("lfp")
    if lfp is None:
        for key in ["data", "signal", "LFP", "Data"]:
            if key in data:
                lfp = data[key]
                break
    if lfp is None:
        raise ValueError(f"No LFP array in {lfp_file}")
    lfp = np.asarray(lfp).squeeze()
    if lfp.ndim > 1:
        lfp = lfp.flatten()
    n_samples = int(lfp.size)
    fs = float(np.squeeze(data.get("fs", 1000.0)))
    return n_samples, fs


def load_ripple_binary_mask(
    bipolar_dir: Path, n_samples: int, use_merged: bool = True
) -> np.ndarray:
    """
    Load ripples_b0xx_zlow3.0.mat and build a binary vector of length n_samples.
    Sets 1 from start to end of each ripple (using merged_starts/merged_ends,
    or real_duration + peak_idx if preferred).
    """
    ripple_file = bipolar_dir / f"ripples_{bipolar_dir.name}_zlow3.0.mat"
    if not ripple_file.exists():
        return np.zeros(n_samples, dtype=np.float64)
    data = sio.loadmat(str(ripple_file), squeeze_me=True)
    binary = np.zeros(n_samples, dtype=np.float64)
    if use_merged and "merged_starts" in data and "merged_ends" in data:
        starts = np.atleast_1d(data["merged_starts"]).astype(np.int64)
        ends = np.atleast_1d(data["merged_ends"]).astype(np.int64)
    else:
        peak_idx = np.atleast_1d(data["peak_idx"]).astype(np.int64)
        rd = np.atleast_2d(data["real_duration"])
        # real_duration is (N,2) with (start_sample, end_sample)
        starts = np.asarray(rd[:, 0], dtype=np.int64)
        ends = np.asarray(rd[:, 1], dtype=np.int64)
    for s, e in zip(starts, ends):
        s, e = int(s), int(e)
        if s < 0:
            s = 0
        if e >= n_samples:
            e = n_samples - 1
        if s <= e:
            binary[s : e + 1] = 1.0
    return binary


def bin_10ms_majority(binary: np.ndarray, bin_samples: int = 10) -> np.ndarray:
    """
    Bin binary vector into 10 ms bins. If more than half of the bin is 1, output 1 else 0.
    bin_samples = 10 at 1 kHz = 10 ms.
    """
    n = len(binary)
    n_bins = n // bin_samples
    trimmed = binary[: n_bins * bin_samples].reshape(n_bins, bin_samples)
    # majority: sum >= half
    half = (bin_samples + 1) // 2
    binned = (trimmed.sum(axis=1) >= half).astype(np.float64)
    return binned


def build_binned_matrix(
    bipolar_dirs: list[Path], n_samples: int, bin_samples: int = 10
) -> np.ndarray:
    """
    Build (n_channels, n_bins) matrix of binned binary ripple series.
    Memory-efficient: one channel binary at a time.
    """
    n_bins = n_samples // bin_samples
    n_ch = len(bipolar_dirs)
    B = np.zeros((n_ch, n_bins), dtype=np.float64)
    for i, d in enumerate(bipolar_dirs):
        binary = load_ripple_binary_mask(d, n_samples)
        B[i] = bin_10ms_majority(binary, bin_samples)
    return B


def cross_corr_lag_fft(
    bi: np.ndarray, bj: np.ndarray, max_lag_bins: int
) -> tuple[float, float]:
    """
    Cross-correlation via FFT; return (lag_bin, max_corr) in range [-max_lag_bins, +max_lag_bins].
    lag_bin > 0 means first signal leads second (bi leads bj).
    """
    n = len(bi)
    # full correlation: length 2*n-1; lag 0 at index n-1
    corr = signal.fftconvolve(bi, bj[::-1].copy(), mode="full")
    # corr indices: 0 -> lag -(n-1), n-1 -> lag 0, 2*n-2 -> lag n-1
    center = n - 1
    lo = max(0, center - max_lag_bins)
    hi = min(len(corr), center + max_lag_bins + 1)
    window = corr[lo:hi]
    if len(window) == 0:
        return 0.0, 0.0
    idx = np.argmax(window)
    lag_bin = idx - (center - lo)
    return float(lag_bin), float(window[idx])


def compute_latency_matrix(
    B: np.ndarray,
    max_lag_bins: int = LAG_WINDOW_BINS,
    bin_ms: float = BIN_MS,
    n_shuffle: int = 0,
    shuffle_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Compute n_ch×n_ch latency matrix (ms) and optional significance mask.

    latency[i,j] = lag in ms: positive means channel i leads channel j.
    If n_shuffle > 0, returns significance_mask (True = significant).
    """
    n_ch = B.shape[0]
    n_bins = B.shape[1]
    latency = np.zeros((n_ch, n_ch), dtype=np.float64)
    max_corr = np.zeros((n_ch, n_ch), dtype=np.float64)
    rng = np.random.default_rng(shuffle_seed)

    for i in range(n_ch):
        for j in range(n_ch):
            if i == j:
                latency[i, j] = 0.0
                max_corr[i, j] = 1.0
                continue
            lag_bin, c = cross_corr_lag_fft(B[i], B[j], max_lag_bins)
            latency[i, j] = lag_bin * bin_ms
            max_corr[i, j] = c

    significance_mask = None
    if n_shuffle > 0:
        # Per-pair null: circular shift channel i, compute max corr in ±lag window for (i,j)
        null_corr = np.zeros((n_ch, n_ch, n_shuffle), dtype=np.float64)
        for s in range(n_shuffle):
            for i in range(n_ch):
                shift = rng.integers(1, n_bins)
                bi_shift = np.roll(B[i], shift)
                for j in range(n_ch):
                    if i == j:
                        null_corr[i, j, s] = 1.0
                        continue
                    _, c = cross_corr_lag_fft(bi_shift, B[j], max_lag_bins)
                    null_corr[i, j, s] = c
        # Significant if observed max_corr > 95th percentile of null for that pair
        threshold = np.percentile(null_corr, 95, axis=2)
        significance_mask = max_corr >= threshold
        for i in range(n_ch):
            significance_mask[i, i] = True

    return latency, significance_mask


def run(
    root: Path,
    session: int,
    trial: int,
    out_dir: Path | None = None,
    n_shuffle: int = 0,
    bin_ms: int = BIN_MS,
    lag_ms: int = LAG_MS,
    max_channels: int | None = None,
    labels_csv: Path | None = None,
    include_regions: list[str] | None = None,
    exclude_regions: list[str] | None = None,
    region_match: str = "either",
    region_match_mode: str = "substring",
    include_channels: list[str] | None = None,
    exclude_channels: list[str] | None = None,
    no_default_rois: bool = False,
    coord_shuffle_null_n: int = 0,
    layout_csv: Path | None = None,
    layout_use_pixels: bool = False,
    coord_shuffle_seed: int = 42,
    coord_shuffle_within_roi_null_n: int = 0,
    roi_labels_for_null: list[str] | None = None,
    roi_label_strategy: str = "roi_prefer",
) -> None:
    trial_dir = root / f"session{session:03d}" / f"trial{trial:03d}_bipolar"
    if out_dir is None:
        out_dir = trial_dir
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bipolar_dirs = get_bipolar_dirs(root, session, trial)
    include_regions = include_regions or []
    exclude_regions = exclude_regions or []
    include_channels = include_channels or []
    exclude_channels = exclude_channels or []

    # Region / channel filtering
    session_dir = root / f"session{session:03d}"
    if labels_csv is None:
        labels_csv = session_dir / "channel_labels.csv"
    channel_labels = load_channel_labels_csv(labels_csv)

    # If the user didn't request any filtering, default to ROI filtering for session 134.
    if (
        session == 134
        and (not no_default_rois)
        and (not include_regions)
        and (not exclude_regions)
        and (not include_channels)
        and (not exclude_channels)
    ):
        include_regions = list(DEFAULT_ROIS_SESSION134)

    if include_regions or exclude_regions or include_channels or exclude_channels:
        if not channel_labels and (include_regions or exclude_regions):
            raise FileNotFoundError(
                f"Region filtering requested but no channel labels CSV found at {labels_csv}"
            )
        bipolar_dirs, _rows = filter_bipolar_dirs(
            bipolar_dirs=bipolar_dirs,
            channel_labels=channel_labels,
            include_regions=include_regions,
            exclude_regions=exclude_regions,
            region_match=region_match,
            region_match_mode=region_match_mode,
            include_channels=include_channels,
            exclude_channels=exclude_channels,
            out_dir=out_dir,
        )
    else:
        # Still emit labels table if available (helps users discover region strings).
        if channel_labels:
            filter_bipolar_dirs(
                bipolar_dirs=bipolar_dirs,
                channel_labels=channel_labels,
                include_regions=[],
                exclude_regions=[],
                region_match="either",
                region_match_mode=region_match_mode,
                include_channels=[],
                exclude_channels=[],
                out_dir=out_dir,
            )

    if max_channels is not None:
        bipolar_dirs = bipolar_dirs[: max_channels]
        print(f"Using first {len(bipolar_dirs)} channels (--max-channels)")
    n_ch = len(bipolar_dirs)
    if n_ch < 2:
        raise ValueError(f"Need at least 2 channels after filtering; got {n_ch}")
    print(f"Found {n_ch} bipolar channels in {trial_dir}")

    n_samples, fs = get_lfp_length_and_fs(bipolar_dirs[0])
    print(f"LFP length: {n_samples} samples, fs={fs} Hz")
    bin_samples = int(round(bin_ms / 1000.0 * fs))
    n_bins = n_samples // bin_samples
    print(f"10 ms binning: {bin_samples} samples/bin -> {n_bins} bins")

    print("Building binned binary ripple matrix...")
    B = build_binned_matrix(bipolar_dirs, n_samples, bin_samples)
    max_lag_bins = int(round(lag_ms / bin_ms))

    print("Computing pairwise lead-lag (FFT cross-correlation)...")
    latency, significance_mask = compute_latency_matrix(
        B, max_lag_bins=max_lag_bins, bin_ms=float(bin_ms), n_shuffle=n_shuffle
    )

    # Save 78×78 latency matrix (ms)
    out_npy = out_dir / "lead_lag_latency_ms.npy"
    np.save(out_npy, latency)
    print(f"Saved latency matrix (ms): {out_npy}")

    channel_names = [d.name for d in bipolar_dirs]
    sio.savemat(
        str(out_dir / "lead_lag_latency_ms.mat"),
        {"latency_ms": latency, "channels": channel_names},
        do_compression=True,
    )
    print(f"Saved latency matrix (mat): {out_dir / 'lead_lag_latency_ms.mat'}")

    # Pairwise table: one row per (i, j), i != j
    pairs_path = out_dir / "lead_lag_pairwise.csv"
    with open(pairs_path, "w") as f:
        f.write("channel_i,channel_j,latency_ms,interpretation\n")
        for i in range(n_ch):
            for j in range(n_ch):
                if i == j:
                    continue
                lat_ms = latency[i, j]
                interp = "i_leads_j" if lat_ms > 0 else ("j_leads_i" if lat_ms < 0 else "synchronous")
                f.write(f"{channel_names[i]},{channel_names[j]},{lat_ms:.2f},{interp}\n")
    print(f"Saved pairwise table: {pairs_path}")

    # Coordinate-shuffle null for distance↔lead–lag relationship (geometry control)
    if coord_shuffle_null_n and int(coord_shuffle_null_n) > 0:
        if layout_csv is None:
            layout_csv = root / "bipolar_layout.csv"
        coords = load_bipolar_layout_coords(layout_csv, channel_names, use_pixels=layout_use_pixels)
        rho_obs, p_naive, distances, flow_mag = compute_distance_flow_correlation(
            latency, channel_names, coords
        )
        null_rhos = coordinate_shuffle_null_for_distance_flow(
            latency,
            channel_names,
            coords,
            n_perm=int(coord_shuffle_null_n),
            seed=int(coord_shuffle_seed),
        )
        # Two-sided empirical p-value
        p_coord_two_sided = float(np.mean(np.abs(null_rhos) >= abs(rho_obs)))

        out_csv = out_dir / "lead_lag_distance_flow_coordshuffle_summary.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "rho_observed",
                    "p_naive_two_sided",
                    "p_coordshuffle_two_sided",
                    "null_mean",
                    "null_median",
                    "null_std",
                    "n_perm",
                    "n_pairs",
                    "layout_csv",
                    "layout_use_pixels",
                ],
            )
            w.writeheader()
            w.writerow(
                {
                    "rho_observed": f"{rho_obs:.6f}",
                    "p_naive_two_sided": f"{p_naive:.6g}",
                    "p_coordshuffle_two_sided": f"{p_coord_two_sided:.6g}",
                    "null_mean": f"{float(np.mean(null_rhos)):.6f}",
                    "null_median": f"{float(np.median(null_rhos)):.6f}",
                    "null_std": f"{float(np.std(null_rhos)):.6f}",
                    "n_perm": int(coord_shuffle_null_n),
                    "n_pairs": int(len(flow_mag)),
                    "layout_csv": str(layout_csv),
                    "layout_use_pixels": int(bool(layout_use_pixels)),
                }
            )
        np.save(out_dir / "lead_lag_coordshuffle_null_rhos.npy", null_rhos)
        sio.savemat(
            str(out_dir / "lead_lag_coordshuffle_null_rhos.mat"),
            {
                "null_rhos": null_rhos,
                "rho_observed": rho_obs,
                "p_naive_two_sided": p_naive,
                "p_coordshuffle_two_sided": p_coord_two_sided,
            },
            do_compression=True,
        )
        print(
            "Coordinate-shuffle null (distance↔|flow|): "
            f"rho_obs={rho_obs:+.3f}, p_coordshuffle(two-sided)={p_coord_two_sided:.4f} "
            f"(n_perm={int(coord_shuffle_null_n)})"
        )

    # Coordinate-shuffle null within each ROI (channel identity ↔ (x,y) randomized within ROI groups)
    if coord_shuffle_within_roi_null_n and int(coord_shuffle_within_roi_null_n) > 0:
        if layout_csv is None:
            layout_csv = root / "bipolar_layout.csv"
        if roi_labels_for_null is None or len(roi_labels_for_null) == 0:
            # Default: session 134 ROIs if available, else fall back to include_regions
            if session == 134 and (not no_default_rois):
                roi_labels_for_null = list(DEFAULT_ROIS_SESSION134)
            else:
                roi_labels_for_null = list(include_regions or [])
        if not roi_labels_for_null:
            raise ValueError(
                "Within-ROI coordinate shuffle requested but no ROI labels provided. "
                "Use --roi-labels-for-null or --include-regions."
            )

        coords = load_bipolar_layout_coords(layout_csv, channel_names, use_pixels=layout_use_pixels)

        # Build ROI mapping for the channels in this run
        # We need the full bipolar_dirs list (same order as channel_names).
        bipolar_to_roi = build_bipolar_to_roi_label(
            bipolar_dirs=bipolar_dirs,
            channel_labels=channel_labels,
            roi_labels=roi_labels_for_null,
            label_strategy=roi_label_strategy,
        )
        missing = [ch for ch in channel_names if ch not in bipolar_to_roi]
        if missing:
            # This is usually due to using non-ROI channels or a label strategy mismatch.
            raise ValueError(
                "Within-ROI coordinate shuffle null requires every analyzed channel to map to an ROI label. "
                f"Missing ROI mapping for {len(missing)}/{len(channel_names)} channels (e.g. {missing[:5]})."
            )

        rho_obs, p_naive, distances, flow_mag = compute_distance_flow_correlation(
            latency, channel_names, coords
        )
        null_rhos = coordinate_shuffle_within_groups_null_for_distance_flow(
            latency,
            channel_names,
            coords,
            channel_to_group=bipolar_to_roi,
            n_perm=int(coord_shuffle_within_roi_null_n),
            seed=int(coord_shuffle_seed),
        )
        p_within_two_sided = float(np.mean(np.abs(null_rhos) >= abs(rho_obs)))

        out_csv = out_dir / "lead_lag_distance_flow_within_roi_coordshuffle_summary.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "rho_observed",
                    "p_naive_two_sided",
                    "p_within_roi_coordshuffle_two_sided",
                    "null_mean",
                    "null_median",
                    "null_std",
                    "n_perm",
                    "n_pairs",
                    "layout_csv",
                    "layout_use_pixels",
                    "roi_label_strategy",
                    "roi_labels",
                ],
            )
            w.writeheader()
            w.writerow(
                {
                    "rho_observed": f"{rho_obs:.6f}",
                    "p_naive_two_sided": f"{p_naive:.6g}",
                    "p_within_roi_coordshuffle_two_sided": f"{p_within_two_sided:.6g}",
                    "null_mean": f"{float(np.mean(null_rhos)):.6f}",
                    "null_median": f"{float(np.median(null_rhos)):.6f}",
                    "null_std": f"{float(np.std(null_rhos)):.6f}",
                    "n_perm": int(coord_shuffle_within_roi_null_n),
                    "n_pairs": int(len(flow_mag)),
                    "layout_csv": str(layout_csv),
                    "layout_use_pixels": int(bool(layout_use_pixels)),
                    "roi_label_strategy": roi_label_strategy,
                    "roi_labels": ",".join(roi_labels_for_null),
                }
            )
        np.save(out_dir / "lead_lag_within_roi_coordshuffle_null_rhos.npy", null_rhos)
        sio.savemat(
            str(out_dir / "lead_lag_within_roi_coordshuffle_null_rhos.mat"),
            {
                "null_rhos": null_rhos,
                "rho_observed": rho_obs,
                "p_naive_two_sided": p_naive,
                "p_within_roi_coordshuffle_two_sided": p_within_two_sided,
            },
            do_compression=True,
        )
        print(
            "Within-ROI coordinate-shuffle null (distance↔|flow|): "
            f"rho_obs={rho_obs:+.3f}, p_within(two-sided)={p_within_two_sided:.4f} "
            f"(n_perm={int(coord_shuffle_within_roi_null_n)})"
        )

    if significance_mask is not None:
        np.save(out_dir / "lead_lag_significance_mask.npy", significance_mask)
        sio.savemat(
            str(out_dir / "lead_lag_significance_mask.mat"),
            {"significance_mask": significance_mask},
            do_compression=True,
        )
        print(f"Saved significance mask (n_shuffle={n_shuffle})")

    return


def main():
    parser = argparse.ArgumentParser(
        description="Lead-Lag analysis on ripple events (78 bipolar channels)"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root path containing session/trial folders",
    )
    parser.add_argument("--session", type=int, default=SESSION, help="Session number")
    parser.add_argument("--trial", type=int, default=TRIAL, help="Trial number")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: trial001_bipolar)",
    )
    parser.add_argument(
        "--n-shuffle",
        type=int,
        default=0,
        help="Number of shuffle iterations for significance (0 = no mask)",
    )
    parser.add_argument(
        "--bin-ms",
        type=float,
        default=BIN_MS,
        help="Binning window in ms",
    )
    parser.add_argument(
        "--lag-window-ms",
        type=float,
        default=LAG_MS,
        help="Lead-lag search window in ms (±)",
    )
    parser.add_argument(
        "--max-channels",
        type=int,
        default=None,
        help="Use only first N channels (for testing)",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=None,
        help="Path to session channel_labels.csv (default: <root>/sessionXXX/channel_labels.csv)",
    )
    parser.add_argument(
        "--include-regions",
        type=str,
        default=None,
        help="Comma-separated region label patterns to include (case-insensitive).",
    )
    parser.add_argument(
        "--exclude-regions",
        type=str,
        default=None,
        help="Comma-separated region label patterns to exclude (case-insensitive).",
    )
    parser.add_argument(
        "--region-match",
        type=str,
        default="either",
        choices=["primary", "either", "both"],
        help="How to match region labels for a bipolar pair: primary=label of orig_ch_a, either=label_a or label_b, both=label_a and label_b.",
    )
    parser.add_argument(
        "--region-match-mode",
        type=str,
        default="substring",
        choices=["substring", "exact"],
        help="Region pattern matching mode (substring or exact).",
    )
    parser.add_argument(
        "--include-channels",
        type=str,
        default=None,
        help="Comma-separated bipolar channels to include (e.g., b001,b002). If set, only these are considered.",
    )
    parser.add_argument(
        "--exclude-channels",
        type=str,
        default=None,
        help="Comma-separated bipolar channels to exclude (e.g., b013,b014).",
    )
    parser.add_argument(
        "--no-default-rois",
        action="store_true",
        help="Disable built-in default ROIs (session 134 only) and run on all channels unless other filters are provided.",
    )
    parser.add_argument(
        "--coord-shuffle-null",
        type=int,
        default=0,
        help="Number of coordinate-shuffle permutations for a geometry-control null on rho(distance, |flow|).",
    )
    parser.add_argument(
        "--layout-csv",
        type=Path,
        default=None,
        help="Path to bipolar_layout.csv (default: <root>/bipolar_layout.csv).",
    )
    parser.add_argument(
        "--layout-use-pixels",
        action="store_true",
        help="Use Xpix/Ypix instead of X/Y from layout CSV.",
    )
    parser.add_argument(
        "--coord-shuffle-seed",
        type=int,
        default=42,
        help="RNG seed for coordinate-shuffle null.",
    )
    parser.add_argument(
        "--coord-shuffle-within-roi-null",
        type=int,
        default=0,
        help="Number of permutations for within-ROI coordinate-shuffle null on rho(distance, |flow|).",
    )
    parser.add_argument(
        "--roi-labels-for-null",
        type=str,
        default=None,
        help="Comma-separated ROI labels to define groups for within-ROI coordinate shuffle (default: session134 ROI list).",
    )
    parser.add_argument(
        "--roi-label-strategy",
        type=str,
        default="roi_prefer",
        choices=["roi_prefer", "a", "b"],
        help="How to assign each bipolar channel to a single ROI label (roi_prefer uses label_a if in ROI else label_b).",
    )
    args = parser.parse_args()
    run(
        root=args.root,
        session=args.session,
        trial=args.trial,
        out_dir=args.out_dir,
        n_shuffle=args.n_shuffle,
        bin_ms=args.bin_ms,
        lag_ms=args.lag_window_ms,
        max_channels=args.max_channels,
        labels_csv=args.labels_csv,
        include_regions=_parse_csv_list(args.include_regions),
        exclude_regions=_parse_csv_list(args.exclude_regions),
        region_match=args.region_match,
        region_match_mode=args.region_match_mode,
        include_channels=_parse_csv_list(args.include_channels),
        exclude_channels=_parse_csv_list(args.exclude_channels),
        no_default_rois=args.no_default_rois,
        coord_shuffle_null_n=args.coord_shuffle_null,
        layout_csv=args.layout_csv,
        layout_use_pixels=args.layout_use_pixels,
        coord_shuffle_seed=args.coord_shuffle_seed,
        coord_shuffle_within_roi_null_n=args.coord_shuffle_within_roi_null,
        roi_labels_for_null=_parse_csv_list(args.roi_labels_for_null),
        roi_label_strategy=args.roi_label_strategy,
    )


if __name__ == "__main__":
    main()
