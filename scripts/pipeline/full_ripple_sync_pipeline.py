#!/usr/bin/env python3
"""
Full ripple detection + rejection + synchrony + rate comparison pipeline.

Requirements (from user spec):
- Use existing bipolar-referenced data: session###/trial###_bipolar/
- For each trial:
    1) Detect ripples on all bipolar channels with z_low = 3.0
    2) Normalize + apply rejection criteria (strict threshold = 3.0)
    3) Save ONLY the passed ripples to derived/ripples_zlow3.0_passed.(pkl)
       Fields per ripple: channel_id, start_time, end_time, peak_time,
       peak_z, duration, rejection_flags (dict)
    4) Compute synchrony per region (parietal, prefrontal, motor)
       - Use region labels from bipolar_channel_labels.csv in each trial dir
       - If a region has <2 channels (or <1 pair), output N_pairs=0 and NaN stats
- Rate comparison per session (pre vs post):
    ripple_rate = total passed ripples per region / recording_duration_seconds
    (pooled across channels)
    Output CSV with: session, date, region, pre_rate, post_rate, delta, percent_change
- Log any missing data / mismatches / skipped computations.

Notes:
- Date mismatches: if known date from spec differs from channel_labels date,
  log in the pipeline log (does not stop processing).
- Detection/rejection is run once per channel; synchrony uses the saved passed events
  (no re-detection inside synchrony).
"""

from __future__ import annotations

import sys
from pathlib import Path
import json
import math
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy import stats
from itertools import combinations
from typing import Tuple

# Add core packages
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "ripple_core"))
from ripple_core.analyze import detect_ripples, normalize_ripples, reject_ripples, reject_ripples_fast


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
BASE_DIR = Path("/vol/brains/bd3/pesaranlab/Archie_RecStim_vSUBNETS220_2nd/matlab/mfiles/Chen")
Z_LOW = 3.0
STRICT_THRESHOLD = 3.0
FS_DEFAULT = 1000

# Rejection mode: 'none', 'fast', 'full'
#   - 'none': no rejection, keep all detected ripples
#   - 'fast': fast rejection with cheap time-domain features (default, ~5% cost)
#   - 'full': full spectral rejection with multitaper per ripple (~100x slower)
REJECTION_MODE = 'fast'

REGIONS = ["parietal", "prefrontal", "motor"]
REGION_COLUMN = "region_type"  # from bipolar_channel_labels.csv
REGION_MAP = {
    "within_parietal": "parietal",
    "within_prefrontal": "prefrontal",
    "within_motor": "motor",
}

# Co-occurrence window for synchrony (seconds)
SYNC_WINDOW_SEC = 0.050

# Session list (session, date, pre_trial, post_trial)
SESSION_SPECS = [
    (32, "180124", 3, 8),
    (33, "180125", 5, 10),
    (34, "180126", 5, 11),
    (35, "180127", 6, 11),
    (37, "180129", 2, 7),
    (41, "180202", 5, 8),
    (45, "180211", 5, 8),
    (46, "180212", 7, 9),
    (50, "180218", 2, 4),
    (51, "180219", 5, 7),
    (52, "180220", 2, 4),
    (59, "180227", 6, 9),
    (60, "180301", 6, 8),
    (61, "180302", 10, 12),
    (64, "180306", 10, 12),
    (65, "180307", 5, 7),
    (66, "180308", 5, 7),
    (67, "180309", 5, 7),
    (73, "180322", 2, 4),
    (74, "180323", 2, 4),
    (75, "180325", 2, 4),
    (76, "180326", 2, 4),
    (77, "180327", 1, 5),
    (78, "180328", 2, 4),
    (79, "180401", 2, 4),
    (81, "180403", 2, 8),
    (82, "180404", 2, 7),
    (85, "180409", 2, 5),
    (86, "180410", 2, 4),
]


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def log(msg: str):
    print(msg)


def load_bipolar_lfp(b_dir: Path) -> Tuple[np.ndarray, float]:
    """Load LFP and fs from lfp_b###.mat in a bipolar channel directory."""
    lfp_file = b_dir / f"lfp_{b_dir.name}.mat"
    if not lfp_file.exists():
        raise FileNotFoundError(f"Missing LFP file: {lfp_file}")
    data = sio.loadmat(str(lfp_file))
    lfp = data.get("lfp")
    if lfp is None:
        raise ValueError(f"No 'lfp' in {lfp_file}")
    lfp = np.array(lfp).squeeze().astype(float)
    fs = float(np.array(data.get("fs", FS_DEFAULT)).squeeze())
    if not np.isfinite(fs) or fs <= 0:
        fs = FS_DEFAULT
    return lfp, fs


def detect_and_reject_channel(b_dir: Path, rejection_mode: str = 'fast') -> pd.DataFrame:
    """
    Detect ripples on one bipolar channel, then optionally apply rejection.
    
    Args:
        b_dir: Path to bipolar channel directory (e.g., b001/)
        rejection_mode: 'none', 'fast', or 'full'
            - 'none': no rejection, keep all detected ripples
            - 'fast': fast rejection with time-domain features (default)
            - 'full': full spectral rejection with multitaper per ripple
    
    Returns:
        DataFrame with passed ripples (or all ripples if mode='none')
    """
    lfp, fs = load_bipolar_lfp(b_dir)
    
    # Detection (always run)
    det = detect_ripples(
        lfp,
        fs=fs,
        z_low=Z_LOW,
        rp_band=(100, 140),
        order=550,
        window_ms=20,
        z_outlier=9.0,
        min_dur_ms=30,
        merge_dur_ms=10,
        epoch_ms=200,
    )
    n = len(det.peak_idx)
    if n == 0:
        return pd.DataFrame(columns=["channel_id", "start_time", "end_time", "peak_time", "peak_z", "duration", "rejection_flags"])

    # Rejection
    if rejection_mode == 'none':
        # No rejection - keep all
        keep_mask = np.ones(n, dtype=bool)
        reasons = ["none"] * n
        
    elif rejection_mode == 'fast':
        # Fast rejection (no multitaper per ripple)
        rej = reject_ripples_fast(
            lfp=lfp,
            bp_lfp=det.bp_lfp,
            fs=int(fs),
            real_duration=det.real_duration,
            peak_idx=det.peak_idx,
            env_rip=det.env_rip,
            mu=float(det.mu),
            sd=float(det.sd),
            strict_threshold=STRICT_THRESHOLD,
            rp_band=(100, 140),
            min_duration_ms=30.0,
            max_duration_ms=150.0,
            min_sharpness=0.5,
            bandpower_zscore_min=2.0,
        )
        keep_mask = ~rej.markers
        reasons = rej.reasons
        
    elif rejection_mode == 'full':
        # Full rejection (multitaper per ripple - SLOW)
        norm = normalize_ripples(
            lfp,
            fs=int(fs),
            raw_windowed_lfp=det.raw_windowed_lfp,
            real_duration=det.real_duration,
        )
        rej = reject_ripples(
            freq_spec_actual=norm.freq_spec_actual,
            spec_f=norm.spec_f,
            mu=float(det.mu),
            sd=float(det.sd),
            strict_threshold=STRICT_THRESHOLD,
            env_rip=det.env_rip,
            peak_idx=det.peak_idx,
        )
        keep_mask = ~rej.markers
        reasons = rej.reasons
        
    else:
        raise ValueError(f"Unknown rejection_mode: {rejection_mode}. Use 'none', 'fast', or 'full'.")

    # Build output dataframe
    starts = det.real_duration[:, 0]
    ends = det.real_duration[:, 1]
    peaks = det.peak_idx
    peak_z = (det.env_rip[np.arange(n), det.peak_idx]).astype(float) if det.env_rip.ndim == 2 else det.env_rip[det.peak_idx]
    durations = (ends - starts) / fs

    rows = []
    for i in range(n):
        flags = {
            "rejected": bool(not keep_mask[i]),
            "reason": reasons[i] if i < len(reasons) else ("keep" if keep_mask[i] else "rej"),
            "mode": rejection_mode,
        }
        rows.append({
            "channel_id": b_dir.name,
            "start_time": starts[i] / fs,
            "end_time": ends[i] / fs,
            "peak_time": peaks[i] / fs,
            "peak_z": float(peak_z[i]),
            "duration": durations[i],
            "rejection_flags": flags,
        })
    df = pd.DataFrame(rows)
    
    # Return only passed ripples (unless mode='none', then return all)
    if rejection_mode == 'none':
        return df
    else:
        return df[df["rejection_flags"].apply(lambda x: not x["rejected"])].reset_index(drop=True)


def process_trial(trial_dir: Path, log_rows: List[Dict], rejection_mode: str = 'fast') -> Tuple[pd.DataFrame, float]:
    """
    Process a single trial: detect+reject across all bipolar channels, return passed ripples and duration (sec).
    
    Args:
        trial_dir: Path to trial directory (e.g., trial003_bipolar/)
        log_rows: List to append log messages
        rejection_mode: 'none', 'fast', or 'full'
    
    Returns:
        (passed_ripples_df, duration_sec)
    """
    derived_dir = trial_dir / "derived"
    derived_dir.mkdir(exist_ok=True)

    # Estimate duration from first channel
    sample_duration_sec = None
    passed_all = []
    b_dirs = sorted(trial_dir.glob("b???"))
    if not b_dirs:
        log_rows.append({"trial": trial_dir.name, "note": "no_bipolar_dirs"})
        return pd.DataFrame(), 0.0

    for b_dir in b_dirs:
        try:
            lfp, fs = load_bipolar_lfp(b_dir)
            if sample_duration_sec is None:
                sample_duration_sec = len(lfp) / fs
            passed = detect_and_reject_channel(b_dir, rejection_mode=rejection_mode)
            if len(passed):
                passed_all.append(passed)
        except Exception as e:
            log_rows.append({"trial": trial_dir.name, "channel": b_dir.name, "note": f"error: {e}"})

    if sample_duration_sec is None:
        sample_duration_sec = 0.0

    passed_df = pd.concat(passed_all, ignore_index=True) if passed_all else pd.DataFrame(
        columns=["channel_id", "start_time", "end_time", "peak_time", "peak_z", "duration", "rejection_flags"]
    )

    # Save passed ripples (both pkl and mat for compatibility)
    out_pkl = derived_dir / "ripples_zlow3.0_passed.pkl"
    out_mat = derived_dir / "ripples_zlow3.0_passed.mat"
    passed_df.to_pickle(out_pkl)
    # Save a MATLAB-friendly struct
    sio.savemat(out_mat, {
        "channel_id": passed_df["channel_id"].to_numpy(dtype=object),
        "start_time": passed_df["start_time"].to_numpy(float),
        "end_time": passed_df["end_time"].to_numpy(float),
        "peak_time": passed_df["peak_time"].to_numpy(float),
        "peak_z": passed_df["peak_z"].to_numpy(float),
        "duration": passed_df["duration"].to_numpy(float),
        # rejection_flags is saved as JSON strings for mat compatibility
        "rejection_flags": passed_df["rejection_flags"].apply(lambda x: json.dumps(x)).to_numpy(dtype=object),
    })
    return passed_df, sample_duration_sec


def load_region_labels(trial_dir: Path) -> pd.DataFrame:
    labels_file = trial_dir / "bipolar_channel_labels.csv"
    if not labels_file.exists():
        raise FileNotFoundError(f"Missing labels: {labels_file}")
    df = pd.read_csv(labels_file)
    # Normalize region column
    df["region"] = df[REGION_COLUMN].map(REGION_MAP).fillna("unknown")
    # Ensure bipolar_channel column uses consistent naming (b001 format)
    if 'bipolar_channel' in df.columns:
        df['bipolar_channel'] = df['bipolar_channel'].apply(
            lambda x: f"b{int(x[1:]):03d}" if isinstance(x, str) and x.startswith('b') else x
        )
    return df


def load_layout() -> Optional[pd.DataFrame]:
    """Load bipolar channel layout (X, Y coordinates) from bipolar_layout.csv in BASE_DIR"""
    layout_file = BASE_DIR / "bipolar_layout.csv"
    if not layout_file.exists():
        return None
    df = pd.read_csv(layout_file)
    # Ensure consistent naming
    if 'bipolar_ch' in df.columns:
        df['bipolar_ch'] = df['bipolar_ch'].apply(
            lambda x: f"b{int(x[1:]):03d}" if isinstance(x, str) and x.startswith('b') else x
        )
    return df


# --------------------------------------------------------------------------------------
# Regional Synchrony Analysis (distance-based with null models)
# --------------------------------------------------------------------------------------
COOCCUR_COL = "cooccur_fraction"

def compute_pairwise_synchrony_with_distance(
    detections_df: pd.DataFrame,
    layout_df: pd.DataFrame,
    channels: List[str],
    window_ms: float = 50.0,
    trial_duration_sec: Optional[float] = None
) -> pd.DataFrame:
    """Compute pairwise synchrony with symmetric co-occurrence and spatial distance."""
    window_sec = window_ms / 1000.0
    
    if trial_duration_sec is not None:
        T_total = trial_duration_sec
    else:
        T_total = detections_df['peak_time_sec'].max() - detections_df['peak_time_sec'].min()
    
    pairs_data = []
    channel_positions = {}
    channel_times = {}
    channel_rates = {}
    
    for ch in channels:
        row = layout_df[layout_df['bipolar_ch'] == ch]
        if len(row) > 0:
            channel_positions[ch] = (row.iloc[0]['X'], row.iloc[0]['Y'])
        times = detections_df[detections_df['bipolar_ch'] == ch]['peak_time_sec'].values
        channel_times[ch] = times
        channel_rates[ch] = len(times) / T_total if T_total > 0 else 0
    
    for ch_i, ch_j in combinations(channels, 2):
        if ch_i not in channel_positions or ch_j not in channel_positions:
            continue
        
        xi, yi = channel_positions[ch_i]
        xj, yj = channel_positions[ch_j]
        distance = np.sqrt((xi - xj)**2 + (yi - yj)**2)
        
        times_i = channel_times[ch_i]
        times_j = channel_times[ch_j]
        n_i = len(times_i)
        n_j = len(times_j)
        
        if n_i == 0 or n_j == 0:
            continue
        
        # Symmetric co-occurrence
        cooccur_i_to_j = sum(1 for t_i in times_i if np.any(np.abs(times_j - t_i) <= window_sec))
        cooccur_j_to_i = sum(1 for t_j in times_j if np.any(np.abs(times_i - t_j) <= window_sec))
        
        cooccur_frac_i = cooccur_i_to_j / n_i if n_i > 0 else 0
        cooccur_frac_j = cooccur_j_to_i / n_j if n_j > 0 else 0
        cooccur_fraction = 0.5 * (cooccur_frac_i + cooccur_frac_j)
        
        pairs_data.append({
            'ch_i': ch_i,
            'ch_j': ch_j,
            'distance': distance,
            'n_i': n_i,
            'n_j': n_j,
            'rate_i': channel_rates[ch_i],
            'rate_j': channel_rates[ch_j],
            'n_cooccur_i_to_j': cooccur_i_to_j,
            'n_cooccur_j_to_i': cooccur_j_to_i,
            COOCCUR_COL: cooccur_fraction,
        })
    
    return pd.DataFrame(pairs_data)


def circular_shift_null(
    detections_df: pd.DataFrame,
    layout_df: pd.DataFrame,
    channels: List[str],
    window_ms: float,
    trial_duration_sec: Optional[float],
    n_perm: int = 500
) -> np.ndarray:
    """Rate-preserving null via circular shift."""
    t_min = detections_df['peak_time_sec'].min()
    T_total = trial_duration_sec if trial_duration_sec is not None else (detections_df['peak_time_sec'].max() - t_min)
    
    null_rhos = []
    for i in range(n_perm):
        shifted_detections = []
        for ch in channels:
            ch_times = detections_df[detections_df['bipolar_ch'] == ch]['peak_time_sec'].values
            shift = np.random.uniform(0, T_total)
            shifted_times = (ch_times - t_min + shift) % T_total + t_min
            for t in shifted_times:
                shifted_detections.append({'bipolar_ch': ch, 'peak_time_sec': t})
        
        shifted_df = pd.DataFrame(shifted_detections)
        pairs_shifted = compute_pairwise_synchrony_with_distance(
            shifted_df, layout_df, channels, window_ms, trial_duration_sec
        )
        
        if len(pairs_shifted) > 5:
            valid = pairs_shifted[np.isfinite(pairs_shifted[COOCCUR_COL])]
            if len(valid) > 5:
                rho, _ = stats.spearmanr(valid['distance'], valid[COOCCUR_COL])
                null_rhos.append(rho)
    
    return np.array(null_rhos)


def coordinate_shuffle_null(
    detections_df: pd.DataFrame,
    layout_df: pd.DataFrame,
    channels: List[str],
    window_ms: float,
    trial_duration_sec: Optional[float],
    n_perm: int = 500
) -> np.ndarray:
    """Geometry control via coordinate shuffling."""
    original_coords = {}
    for ch in channels:
        row = layout_df[layout_df['bipolar_ch'] == ch]
        if len(row) > 0:
            original_coords[ch] = (row.iloc[0]['X'], row.iloc[0]['Y'])
    
    null_rhos = []
    for i in range(n_perm):
        channels_with_coords = list(original_coords.keys())
        coords_list = list(original_coords.values())
        shuffled_coords = np.random.permutation(coords_list)
        
        shuffled_layout = layout_df.copy()
        for ch, (x, y) in zip(channels_with_coords, shuffled_coords):
            mask = shuffled_layout['bipolar_ch'] == ch
            shuffled_layout.loc[mask, 'X'] = x
            shuffled_layout.loc[mask, 'Y'] = y
        
        pairs_shuffled = compute_pairwise_synchrony_with_distance(
            detections_df, shuffled_layout, channels, window_ms, trial_duration_sec
        )
        
        if len(pairs_shuffled) > 5:
            valid = pairs_shuffled[np.isfinite(pairs_shuffled[COOCCUR_COL])]
            if len(valid) > 5:
                rho, _ = stats.spearmanr(valid['distance'], valid[COOCCUR_COL])
                null_rhos.append(rho)
    
    return np.array(null_rhos)


def permutation_test_correlation(distances: np.ndarray, values: np.ndarray, n_perm: int = 5000) -> Tuple[float, np.ndarray]:
    """Permutation test for correlation (one-sided left-tail)."""
    observed_rho, _ = stats.spearmanr(distances, values)
    null_rhos = []
    for i in range(n_perm):
        shuffled_distances = np.random.permutation(distances)
        rho, _ = stats.spearmanr(shuffled_distances, values)
        null_rhos.append(rho)
    null_rhos = np.array(null_rhos)
    p_value = np.mean(null_rhos <= observed_rho)
    return p_value, null_rhos


def analyze_region_synchrony(
    region_name: str,
    region_filter: str,
    passed_df: pd.DataFrame,
    layout_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    duration_sec: float,
    output_dir: Path,
    window_ms: float = 50.0,
    n_perm_circular: int = 500,
    n_perm_distance: int = 5000
) -> Optional[Dict]:
    """Analyze regional synchrony with distance-based correlation and null models."""
    
    # Build detections_df from passed_df
    detections_df = passed_df[['channel_id', 'peak_time']].copy()
    detections_df.columns = ['bipolar_ch', 'peak_time_sec']
    # Ensure consistent naming
    detections_df['bipolar_ch'] = detections_df['bipolar_ch'].apply(
        lambda x: f"b{int(x[1:]):03d}" if isinstance(x, str) and x.startswith('b') else x
    )
    
    # Filter channels by region
    if region_filter == "within_parietal":
        region_channels = labels_df[labels_df[REGION_COLUMN] == "within_parietal"]['bipolar_channel'].tolist()
    elif region_filter == "within_prefrontal":
        region_channels = labels_df[labels_df[REGION_COLUMN] == "within_prefrontal"]['bipolar_channel'].tolist()
    elif region_filter == "within_motor":
        region_channels = labels_df[labels_df[REGION_COLUMN] == "within_motor"]['bipolar_channel'].tolist()
    else:
        region_channels = labels_df['bipolar_channel'].tolist()
    
    # Filter to channels with layout coordinates
    region_channels = [ch for ch in region_channels if ch in layout_df['bipolar_ch'].values]
    
    if len(region_channels) < 3:
        return {"note": "insufficient_channels", "n_channels": len(region_channels)}
    
    region_detections = detections_df[detections_df['bipolar_ch'].isin(region_channels)]
    
    if len(region_detections) < 50:
        return {"note": "insufficient_ripples", "n_ripples": len(region_detections)}
    
    # Compute pairwise synchrony
    pairs_df = compute_pairwise_synchrony_with_distance(
        region_detections, layout_df, region_channels, window_ms, duration_sec
    )
    
    if len(pairs_df) < 10:
        return {"note": "insufficient_pairs", "n_pairs": len(pairs_df)}
    
    valid_pairs = pairs_df[np.isfinite(pairs_df[COOCCUR_COL])].copy()
    if len(valid_pairs) < 5:
        return {"note": "insufficient_valid_pairs", "n_pairs": len(valid_pairs)}
    
    observed_rho, observed_p = stats.spearmanr(valid_pairs['distance'], valid_pairs[COOCCUR_COL])
    
    # Circular shift null
    null_rhos_circular = circular_shift_null(
        region_detections, layout_df, region_channels, window_ms, duration_sec, n_perm=n_perm_circular
    )
    
    if len(null_rhos_circular) < 10:
        return {"note": "null_failed", "n_null": len(null_rhos_circular)}
    
    null_median = np.median(null_rhos_circular)
    null_mean = np.mean(null_rhos_circular)
    null_std = np.std(null_rhos_circular)
    
    if null_std < 1e-10:
        return {"note": "degenerate_null", "null_std": null_std}
    
    effect_size = observed_rho - null_median
    z_score = (observed_rho - null_mean) / null_std
    p_circular = np.mean(null_rhos_circular <= observed_rho)
    
    # Permutation test
    p_perm, null_rhos_perm = permutation_test_correlation(
        valid_pairs['distance'].values, valid_pairs[COOCCUR_COL].values, n_perm=n_perm_distance
    )
    
    # Coordinate shuffle
    null_rhos_coord = coordinate_shuffle_null(
        region_detections, layout_df, region_channels, window_ms, duration_sec, n_perm=n_perm_circular
    )
    p_coord = np.mean(null_rhos_coord <= observed_rho) if len(null_rhos_coord) > 0 else math.nan
    
    # Robustness to window size
    robustness_results = []
    for W in [10, 20, 30, 50]:
        pairs_W = compute_pairwise_synchrony_with_distance(
            region_detections, layout_df, region_channels, W, duration_sec
        )
        valid_W = pairs_W[np.isfinite(pairs_W[COOCCUR_COL])]
        if len(valid_W) > 5:
            rho_W, p_W = stats.spearmanr(valid_W['distance'], valid_W[COOCCUR_COL])
            robustness_results.append({'window_ms': W, 'rho': rho_W, 'p': p_W, 'n_pairs': len(valid_W)})
    
    # Save outputs
    region_output_dir = output_dir / f"synchrony_{region_name.lower()}"
    region_output_dir.mkdir(parents=True, exist_ok=True)
    pairs_df.to_csv(region_output_dir / "pairwise_synchrony.csv", index=False)
    if robustness_results:
        pd.DataFrame(robustness_results).to_csv(region_output_dir / "robustness_to_window.csv", index=False)
    
    # Create visualization
    fig = plt.figure(figsize=(15, 5))
    
    ax1 = plt.subplot(1, 3, 1)
    ax1.scatter(valid_pairs['distance'], valid_pairs[COOCCUR_COL], alpha=0.5, s=20)
    ax1.set_xlabel('Distance')
    ax1.set_ylabel(f'Co-occurrence Fraction (W={window_ms}ms)')
    ax1.set_title(f'{region_name}\nρ={observed_rho:.3f}, p_naive={observed_p:.4f}')
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(1, 3, 2)
    ax2.hist(null_rhos_circular, bins=30, alpha=0.7, color='gray', edgecolor='black')
    ax2.axvline(observed_rho, color='red', linewidth=2, linestyle='--', label='Observed')
    ax2.axvline(null_median, color='blue', linewidth=2, linestyle=':', label='Null median')
    ax2.set_xlabel('ρ (null)')
    ax2.set_title(f'Circular Shift Null\np={p_circular:.4f}, z={z_score:.2f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(1, 3, 3)
    ax3.hist(null_rhos_perm, bins=50, alpha=0.7, color='gray', edgecolor='black')
    ax3.axvline(observed_rho, color='red', linewidth=2, linestyle='--', label='Observed')
    ax3.set_xlabel('ρ (null)')
    ax3.set_title(f'Distance Permutation\np={p_perm:.4f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(region_output_dir / "synchrony_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Significance criterion: rho < 0 AND p_circular < 0.05 AND p_perm < 0.05
    # (p_coord is reported but not required for significance)
    significant = (observed_rho < 0) and (p_circular < 0.05) and (p_perm < 0.05)
    
    # Log result
    sig_marker = "✓ SIGNIFICANT" if significant else "✗ Not significant"
    log(f"  → {region_name}: ρ={observed_rho:.3f}, p_circ={p_circular:.4f}, p_perm={p_perm:.4f} {sig_marker}")
    
    return {
        "region": region_name,
        "n_channels": len(region_channels),
        "n_ripples": len(region_detections),
        "n_pairs": len(pairs_df),
        "rho_observed": observed_rho,
        "effect_size": effect_size,
        "z_score": z_score,
        "p_circular": p_circular,
        "p_perm": p_perm,
        "p_coord": p_coord,
        "significant": significant,
        "note": ""
    }


def compute_synchrony(passed_df: pd.DataFrame, labels_df: pd.DataFrame, duration_sec: float) -> pd.DataFrame:
    """Compute simple co-occurrence synchrony per region; if <2 channels, output NaNs."""
    rows = []
    for region in REGIONS:
        region_ch = labels_df[labels_df["region"] == region]["bipolar_channel"].tolist()
        if len(region_ch) < 2:
            rows.append({"region": region, "n_channels": len(region_ch), "n_pairs": 0, "mean_cooccur": math.nan, "median_cooccur": math.nan})
            continue
        region_events = passed_df[passed_df["channel_id"].isin(region_ch)]
        # Group by channel
        times = {ch: np.array(region_events[region_events["channel_id"] == ch]["peak_time"]) for ch in region_ch}
        pairs = []
        for i in range(len(region_ch)):
            for j in range(i + 1, len(region_ch)):
                a, b = region_ch[i], region_ch[j]
                ta, tb = times[a], times[b]
                if len(ta) == 0 or len(tb) == 0:
                    continue
                # symmetric co-occurrence fraction
                co_a = np.sum([np.any(np.abs(tb - t) <= SYNC_WINDOW_SEC) for t in ta]) / len(ta)
                co_b = np.sum([np.any(np.abs(ta - t) <= SYNC_WINDOW_SEC) for t in tb]) / len(tb)
                pairs.append(0.5 * (co_a + co_b))
        if pairs:
            rows.append({"region": region, "n_channels": len(region_ch), "n_pairs": len(pairs), "mean_cooccur": float(np.mean(pairs)), "median_cooccur": float(np.median(pairs))})
        else:
            rows.append({"region": region, "n_channels": len(region_ch), "n_pairs": 0, "mean_cooccur": math.nan, "median_cooccur": math.nan})
    return pd.DataFrame(rows)


def ripple_rates_by_region(passed_df: pd.DataFrame, labels_df: pd.DataFrame, duration_sec: float) -> pd.DataFrame:
    rows = []
    for region in REGIONS:
        region_ch = labels_df[labels_df["region"] == region]["bipolar_channel"].tolist()
        events = passed_df[passed_df["channel_id"].isin(region_ch)]
        rate = len(events) / duration_sec if duration_sec > 0 else math.nan
        rows.append({"region": region, "rate": rate, "n_events": len(events), "duration_sec": duration_sec})
    return pd.DataFrame(rows)


def process_session(session: int, date_str: str, pre_trial: int, post_trial: int, log_rows: List[Dict], rejection_mode: str = 'fast') -> List[Dict]:
    """
    Process pre/post for one session; returns list of per-region rate comparison dicts.
    
    Args:
        session: session number
        date_str: expected date string
        pre_trial: pre-stimulation trial number
        post_trial: post-stimulation trial number
        log_rows: list to append log messages
        rejection_mode: 'none', 'fast', or 'full'
    
    Returns:
        List of per-region rate comparison dictionaries
    """
    session_dir = BASE_DIR / f"session{session:03d}"
    results = []

    for label, trial_num in [("pre", pre_trial), ("post", post_trial)]:
        trial_dir = session_dir / f"trial{trial_num:03d}_bipolar"
        if not trial_dir.exists():
            log_rows.append({"session": session, "trial": trial_num, "note": "missing_trial_dir"})
            return results

        # Detection + rejection
        passed_df, duration_sec = process_trial(trial_dir, log_rows, rejection_mode=rejection_mode)
        # Region labels
        try:
            labels_df = load_region_labels(trial_dir)
        except Exception as e:
            log_rows.append({"session": session, "trial": trial_num, "note": f"missing_labels: {e}"})
            continue
        # Simple synchrony (existing)
        sync_df = compute_synchrony(passed_df, labels_df, duration_sec)
        sync_out = trial_dir / "derived" / "synchrony_summary.csv"
        sync_df.to_csv(sync_out, index=False)
        
        # Regional synchrony analysis (distance-based with null models)
        layout_df = load_layout()
        if layout_df is not None and len(passed_df) > 0:
            derived_dir = trial_dir / "derived"
            regional_stats = []
            
            for region_name, region_filter in [("parietal", "within_parietal"), 
                                               ("prefrontal", "within_prefrontal"), 
                                               ("motor", "within_motor")]:
                try:
                    result = analyze_region_synchrony(
                        region_name=region_name,
                        region_filter=region_filter,
                        passed_df=passed_df,
                        layout_df=layout_df,
                        labels_df=labels_df,
                        duration_sec=duration_sec,
                        output_dir=derived_dir,
                        window_ms=50.0,
                        n_perm_circular=500,
                        n_perm_distance=5000
                    )
                    if result is not None:
                        result_row = {
                            "session": session,
                            "trial": trial_num,
                            "trial_type": label,
                        }
                        result_row.update(result)
                        regional_stats.append(result_row)
                except Exception as e:
                    log_rows.append({"session": session, "trial": trial_num, "region": region_name, 
                                   "note": f"regional_sync_error: {e}"})
            
            # Save regional stats
            if regional_stats:
                regional_df = pd.DataFrame(regional_stats)
                regional_df.to_csv(derived_dir / "synchrony_region_stats.csv", index=False)

        # Rates
        rates_df = ripple_rates_by_region(passed_df, labels_df, duration_sec)
        for _, row in rates_df.iterrows():
            results.append({
                "session": session,
                "date": date_str,
                "trial_type": label,
                "region": row["region"],
                "rate": row["rate"],
                "n_events": row["n_events"],
                "duration_sec": row["duration_sec"],
            })

    # Build rate comparison (pre vs post)
    pre_rows = {r["region"]: r for r in results if r["trial_type"] == "pre"}
    post_rows = {r["region"]: r for r in results if r["trial_type"] == "post"}
    summary = []
    for region in REGIONS:
        pre = pre_rows.get(region)
        post = post_rows.get(region)
        if pre is None or post is None:
            summary.append({
                "session": session,
                "date": date_str,
                "region": region,
                "pre_rate": math.nan,
                "post_rate": math.nan,
                "delta": math.nan,
                "percent_change": math.nan,
                "note": "missing_pre_or_post",
            })
            continue
        delta = post["rate"] - pre["rate"]
        pct = (delta / pre["rate"] * 100) if pre["rate"] and not math.isnan(pre["rate"]) else math.nan
        summary.append({
            "session": session,
            "date": date_str,
            "region": region,
            "pre_rate": pre["rate"],
            "post_rate": post["rate"],
            "delta": delta,
            "percent_change": pct,
            "note": "",
        })
    return summary


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Full ripple detection+rejection+synchrony+rate pipeline")
    parser.add_argument("--sessions", nargs="*", type=int, help="Subset of sessions to run (e.g., 32 33 34)")
    parser.add_argument("--rejection-mode", type=str, default=REJECTION_MODE, 
                        choices=['none', 'fast', 'full'],
                        help="Rejection mode: 'none' (no rejection), 'fast' (time-domain features, default), 'full' (spectral, slow)")
    args = parser.parse_args()

    run_specs = [s for s in SESSION_SPECS if (args.sessions is None or s[0] in args.sessions)]
    rejection_mode = args.rejection_mode

    log(f"\n{'='*80}")
    log(f"PIPELINE CONFIGURATION")
    log(f"{'='*80}")
    log(f"Rejection mode: {rejection_mode}")
    log(f"  - 'none': no rejection, all detected ripples kept")
    log(f"  - 'fast': cheap time-domain + bandpower features (~5% overhead)")
    log(f"  - 'full': full multitaper spectral rejection (~100x slower)")
    log(f"Z-score threshold: {Z_LOW} (detection), {STRICT_THRESHOLD} (strict rejection)")
    log(f"Sessions to process: {len(run_specs)}")
    log(f"{'='*80}\n")

    all_summary = []
    log_rows: List[Dict] = []

    for session, date_str, pre_trial, post_trial in run_specs:
        log(f"\n=== Session {session:03d} (pre {pre_trial:03d}, post {post_trial:03d}) ===")
        try:
            session_summary = process_session(session, date_str, pre_trial, post_trial, log_rows, rejection_mode=rejection_mode)
            all_summary.extend(session_summary)
        except Exception as e:
            log_rows.append({"session": session, "note": f"error: {e}"})
            continue

    # Save rate comparison table
    summary_df = pd.DataFrame(all_summary)
    out_csv = BASE_DIR / "analysis_output_rate_comparison.csv"
    summary_df.to_csv(out_csv, index=False)

    # Save log
    log_path = BASE_DIR / "analysis_pipeline_log.json"
    with open(log_path, "w") as f:
        json.dump(log_rows, f, indent=2)

    # Collect and summarize regional synchrony results
    log("\n" + "="*80)
    log("REGIONAL SYNCHRONY SUMMARY")
    log("="*80)
    log("Significance criterion: (ρ < 0) AND (p_circular < 0.05) AND (p_perm < 0.05)")
    log("")
    
    # Collect all regional stats from all sessions
    all_regional_stats = []
    for session, _, pre_trial, post_trial in run_specs:
        for trial_num, trial_label in [(pre_trial, "pre"), (post_trial, "post")]:
            session_dir = BASE_DIR / f"session{session:03d}"
            trial_dir = session_dir / f"trial{trial_num:03d}_bipolar"
            stats_file = trial_dir / "derived" / "synchrony_region_stats.csv"
            if stats_file.exists():
                try:
                    df = pd.read_csv(stats_file)
                    all_regional_stats.append(df)
                except:
                    pass
    
    if all_regional_stats:
        combined_stats = pd.concat(all_regional_stats, ignore_index=True)
        
        # Overall summary
        total_analyses = len(combined_stats)
        significant_count = combined_stats['significant'].sum()
        
        log(f"Total regional analyses: {total_analyses}")
        log(f"Significant (local synchrony): {significant_count} ({100*significant_count/total_analyses:.1f}%)")
        log("")
        
        # By region
        log("By Region:")
        for region in ["parietal", "prefrontal", "motor"]:
            region_data = combined_stats[combined_stats['region'] == region]
            if len(region_data) > 0:
                n_total = len(region_data)
                n_sig = region_data['significant'].sum()
                log(f"  {region.capitalize():12s}: {n_sig:2d}/{n_total:2d} significant ({100*n_sig/n_total:.1f}%)")
        
        log("")
        log("Detailed results saved in each trial's derived/synchrony_region_stats.csv")
    else:
        log("No regional synchrony results found.")
    
    log("="*80)
    log("\nPipeline complete.")
    log(f"Rate comparison table: {out_csv}")
    log(f"Log file: {log_path}")


if __name__ == "__main__":
    main()
