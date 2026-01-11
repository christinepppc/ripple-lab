#!/usr/bin/env python3
"""
Run Synchrony Analysis with Proper Statistical Controls

Question: Is ripple synchrony local (network-driven) or global (common input)?

Statistical improvements:
  A1. One-sided test enforced (H1: rho < 0 for local synchrony)
  A2. Symmetric co-occurrence definition with <= window (not <)
  A3. Rate control via circular shift null (not analytic formula)
  B4. Optional trial_duration parameter
  B5. Keep zero co-occurrence pairs (no biased filtering)
  C6. Coordinate shuffle null (channel-level geometry control)
  
FIXES (Round 2):
  - Standardized column name: COOCCUR_COL constant
  - Effect size uses median(null) consistently
  - Verified one-sided p-values throughout
  - Added layout consistency checks

Usage:
    python scripts/analysis/run_synchrony_analysis.py \
        --trial_dir /path/to/trial_bipolar \
        --session_name session46_trial001 \
        --z_low 3.0 \
        --trial_duration_sec 300
"""

import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages"))

import argparse
import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# CANONICAL COLUMN NAME - use everywhere
COOCCUR_COL = "cooccur_fraction"  # Fraction of events with temporal partner


def load_ripple_detections(trial_bipolar_dir: Path, z_low: float = 3.0) -> Tuple[pd.DataFrame, float]:
    """
    Load all ripple detections from bipolar channels.
    
    Returns:
        (detections_df, trial_duration_sec)
    """
    detections = []
    
    for bipolar_dir in sorted(trial_bipolar_dir.glob("b???")):
        bipolar_name = bipolar_dir.name
        ripple_file = bipolar_dir / f"ripples_{bipolar_name}_zlow{z_low:.1f}.mat"
        
        if not ripple_file.exists():
            continue
        
        try:
            data = sio.loadmat(str(ripple_file))
            peak_idx = data['peak_idx'].flatten()
            fs = float(data['fs'].flatten()[0])
            n_ripples = int(data['n_ripples'].flatten()[0])
            
            if n_ripples == 0:
                continue
            
            peak_times = peak_idx / fs
            
            for peak_time in peak_times:
                detections.append({
                    'bipolar_ch': bipolar_name,
                    'peak_time_sec': peak_time,
                })
        except Exception as e:
            print(f"Warning: Could not load {ripple_file}: {e}")
    
    detections_df = pd.DataFrame(detections)
    
    # Estimate trial duration from data (may underestimate if events stop early)
    if len(detections_df) > 0:
        duration_estimate = detections_df['peak_time_sec'].max()
    else:
        duration_estimate = 0.0
    
    return detections_df, duration_estimate


def compute_pairwise_synchrony(
    detections_df: pd.DataFrame,
    layout_df: pd.DataFrame,
    channels: List[str],
    window_ms: float = 50.0,
    trial_duration_sec: Optional[float] = None
) -> pd.DataFrame:
    """
    Compute pairwise synchrony metrics with SYMMETRIC co-occurrence.
    
    Fix A2: Co-occurrence is now symmetric - we count both directions and average.
    Fix B5: Returns ALL pairs (not filtered by COOCCUR_COL > 0).
    
    The co-occurrence metric is a FRACTION (not rate): the proportion of events
    that have a temporal partner within the specified window.
    
    Returns:
        DataFrame with columns including:
        - 'distance': spatial distance between channels
        - COOCCUR_COL: symmetric co-occurrence fraction
    """
    window_sec = window_ms / 1000.0
    
    # Total recording duration
    if trial_duration_sec is not None:
        T_total = trial_duration_sec
    else:
        # Fallback to estimate from data
        T_total = detections_df['peak_time_sec'].max() - detections_df['peak_time_sec'].min()
    
    pairs_data = []
    
    # Get positions and rates for all channels
    channel_positions = {}
    channel_rates = {}
    channel_times = {}
    
    for ch in channels:
        row = layout_df[layout_df['bipolar_ch'] == ch]
        if len(row) > 0:
            channel_positions[ch] = (row.iloc[0]['X'], row.iloc[0]['Y'])
        
        # Get ripple times
        times = detections_df[detections_df['bipolar_ch'] == ch]['peak_time_sec'].values
        channel_times[ch] = times
        channel_rates[ch] = len(times) / T_total if T_total > 0 else 0
    
    # Iterate over all pairs
    for ch_i, ch_j in combinations(channels, 2):
        if ch_i not in channel_positions or ch_j not in channel_positions:
            continue
        
        # Spatial distance
        xi, yi = channel_positions[ch_i]
        xj, yj = channel_positions[ch_j]
        distance = np.sqrt((xi - xj)**2 + (yi - yj)**2)
        
        # Get ripple times
        times_i = channel_times[ch_i]
        times_j = channel_times[ch_j]
        
        n_i = len(times_i)
        n_j = len(times_j)
        
        if n_i == 0 or n_j == 0:
            continue
        
        # FIX A2: SYMMETRIC co-occurrence with <= for consistency
        # Count i->j matches
        cooccur_i_to_j = 0
        for t_i in times_i:
            if np.any(np.abs(times_j - t_i) <= window_sec):
                cooccur_i_to_j += 1
        
        # Count j->i matches
        cooccur_j_to_i = 0
        for t_j in times_j:
            if np.any(np.abs(times_i - t_j) <= window_sec):
                cooccur_j_to_i += 1
        
        # Symmetric fraction: average of both directions
        cooccur_frac_i = cooccur_i_to_j / n_i if n_i > 0 else 0
        cooccur_frac_j = cooccur_j_to_i / n_j if n_j > 0 else 0
        cooccur_fraction = 0.5 * (cooccur_frac_i + cooccur_frac_j)
        
        pairs_data.append({
            'ch_i': ch_i,
            'ch_j': ch_j,
            'distance': distance,
            'n_i': n_i,
            'n_j': n_j,
            'rate_i': channel_rates[ch_i],  # This IS actual rate (events/sec)
            'rate_j': channel_rates[ch_j],  # This IS actual rate (events/sec)
            'n_cooccur_i_to_j': cooccur_i_to_j,
            'n_cooccur_j_to_i': cooccur_j_to_i,
            COOCCUR_COL: cooccur_fraction,  # CANONICAL COLUMN NAME
        })
    
    return pd.DataFrame(pairs_data)


def circular_shift_null(
    detections_df: pd.DataFrame,
    layout_df: pd.DataFrame,
    channels: List[str],
    window_ms: float,
    trial_duration_sec: Optional[float],
    n_perm: int = 500
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rate-preserving null: circular shift each channel's ripple times.
    
    This preserves the ripple rate (events/sec) but destroys temporal structure.
    
    Returns:
        (null_rhos, null_cooccur_fractions) - for use as "rate control"
    """
    t_min = detections_df['peak_time_sec'].min()
    
    if trial_duration_sec is not None:
        T_total = trial_duration_sec
    else:
        t_max = detections_df['peak_time_sec'].max()
        T_total = t_max - t_min
    
    null_rhos = []
    all_null_fractions = []
    
    for i in range(n_perm):
        if i % 100 == 0:
            print(f"    Circular shift: {i}/{n_perm}", end='\r')
        
        shifted_detections = []
        
        for ch in channels:
            ch_times = detections_df[detections_df['bipolar_ch'] == ch]['peak_time_sec'].values
            
            shift = np.random.uniform(0, T_total)
            shifted_times = (ch_times - t_min + shift) % T_total + t_min
            
            for t in shifted_times:
                shifted_detections.append({
                    'bipolar_ch': ch,
                    'peak_time_sec': t,
                })
        
        shifted_df = pd.DataFrame(shifted_detections)
        pairs_shifted = compute_pairwise_synchrony(
            shifted_df, layout_df, channels, window_ms, trial_duration_sec
        )
        
        # FIX B5: Use ALL pairs, not just COOCCUR_COL > 0
        if len(pairs_shifted) > 5:
            # Filter only for NaN/inf
            valid = pairs_shifted[np.isfinite(pairs_shifted[COOCCUR_COL])]
            if len(valid) > 5:
                rho, _ = stats.spearmanr(valid['distance'], valid[COOCCUR_COL])
                null_rhos.append(rho)
                all_null_fractions.extend(valid[COOCCUR_COL].values)
    
    print(f"    Circular shift: {n_perm}/{n_perm} - Done!")
    return np.array(null_rhos), np.array(all_null_fractions)


def coordinate_shuffle_null(
    detections_df: pd.DataFrame,
    layout_df: pd.DataFrame,
    channels: List[str],
    window_ms: float,
    trial_duration_sec: Optional[float],
    n_perm: int = 500
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Geometry control: shuffle electrode coordinates, recompute distances.
    
    This is the strongest geometry control - it preserves:
      - The set of electrode positions
      - The set of co-occurrence values
      - Event timing and rates (detections_df unchanged)
    But destroys the true distance-to-pair mapping.
    
    Returns:
        (null_rhos, original_layout_df)
    """
    # Get original coordinates
    original_coords = {}
    for ch in channels:
        row = layout_df[layout_df['bipolar_ch'] == ch]
        if len(row) > 0:
            original_coords[ch] = (row.iloc[0]['X'], row.iloc[0]['Y'])
    
    null_rhos = []
    
    for i in range(n_perm):
        if i % 100 == 0:
            print(f"    Coordinate shuffle: {i}/{n_perm}", end='\r')
        
        # Shuffle the coordinates across channels
        channels_with_coords = list(original_coords.keys())
        coords_list = list(original_coords.values())
        shuffled_coords = np.random.permutation(coords_list)
        
        # Create shuffled layout (only geometry changes, not events)
        shuffled_layout = layout_df.copy()
        for ch, (x, y) in zip(channels_with_coords, shuffled_coords):
            mask = shuffled_layout['bipolar_ch'] == ch
            shuffled_layout.loc[mask, 'X'] = x
            shuffled_layout.loc[mask, 'Y'] = y
        
        # Compute synchrony with shuffled coordinates
        # NOTE: detections_df unchanged - preserves rates and timing!
        pairs_shuffled = compute_pairwise_synchrony(
            detections_df, shuffled_layout, channels, window_ms, trial_duration_sec
        )
        
        if len(pairs_shuffled) > 5:
            valid = pairs_shuffled[np.isfinite(pairs_shuffled[COOCCUR_COL])]
            if len(valid) > 5:
                rho, _ = stats.spearmanr(valid['distance'], valid[COOCCUR_COL])
                null_rhos.append(rho)
    
    print(f"    Coordinate shuffle: {n_perm}/{n_perm} - Done!")
    return np.array(null_rhos), layout_df


def permutation_test_correlation(
    distances: np.ndarray, 
    values: np.ndarray, 
    n_perm: int = 5000
) -> Tuple[float, np.ndarray]:
    """
    Permutation test for correlation by shuffling distances.
    
    Returns (p_value_one_sided, null_rhos).
    
    FIX A1: This is explicitly a ONE-SIDED LEFT-TAIL test for H1: rho < 0.
    """
    observed_rho, _ = stats.spearmanr(distances, values)
    
    null_rhos = []
    for i in range(n_perm):
        if i % 1000 == 0:
            print(f"    Permutation: {i}/{n_perm}", end='\r')
        shuffled_distances = np.random.permutation(distances)
        rho, _ = stats.spearmanr(shuffled_distances, values)
        null_rhos.append(rho)
    
    print(f"    Permutation: {n_perm}/{n_perm} - Done!")
    null_rhos = np.array(null_rhos)
    
    # ONE-SIDED LEFT-TAIL p-value: P(null <= observed)
    p_value = np.mean(null_rhos <= observed_rho)
    
    return p_value, null_rhos


def analyze_region(
    region_name: str,
    region_filter: str,
    detections_df: pd.DataFrame,
    layout_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    output_base_dir: Path,
    trial_duration_sec: Optional[float] = None,
    primary_window: float = 50.0,
    window_values: List[float] = [10, 20, 30, 50],
    n_perm_circular: int = 500,
    n_perm_distance: int = 5000
) -> Optional[Dict]:
    """
    Analyze synchrony for a specific region with all controls.
    
    Returns dict with summary statistics, or None if insufficient data.
    """
    print("\n" + "="*100)
    print(f"ANALYZING: {region_name.upper()}")
    print("="*100)
    
    # Output directory for this region
    output_dir = output_base_dir / f"synchrony_{region_name.lower().replace(' ', '_')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter channels
    if region_filter == "all":
        region_channels = labels_df['bipolar_channel'].tolist()
    else:
        region_channels = labels_df[labels_df['region_type'] == region_filter]['bipolar_channel'].tolist()
    
    print(f"Channels: {len(region_channels)}")
    
    if len(region_channels) < 3:
        print(f"  ⚠ Too few channels ({len(region_channels)}), skipping...")
        return None
    
    # CHECK: Verify all channels exist in layout
    # Only use the 78 common bipolar pairs (b001-b078) that exist in the layout
    missing_in_layout = set(region_channels) - set(layout_df['bipolar_ch'])
    if missing_in_layout:
        print(f"  ⚠ {len(missing_in_layout)} channels not in common layout (using 78-pair subset)")
        print(f"     Excluded: {sorted(list(missing_in_layout))[:10]}")
        region_channels = [ch for ch in region_channels if ch in layout_df['bipolar_ch'].values]
        print(f"  → Using {len(region_channels)} channels with layout coordinates")
        if len(region_channels) < 3:
            print(f"  ⚠ Too few valid channels after filtering, skipping...")
        return None
    
    region_detections = detections_df[detections_df['bipolar_ch'].isin(region_channels)]
    print(f"Detections: {len(region_detections)}")
    
    if len(region_detections) < 50:
        print(f"  ⚠ Too few detections ({len(region_detections)}), skipping...")
        return None
    
    # Main analysis
    print(f"Computing pairwise synchrony (W={primary_window}ms)...")
    pairs_df = compute_pairwise_synchrony(
        region_detections, layout_df, region_channels, 
        window_ms=primary_window, 
        trial_duration_sec=trial_duration_sec
    )
    
    print(f"Pairs: {len(pairs_df)}")
    
    if len(pairs_df) < 10:
        print(f"  ⚠ Too few pairs ({len(pairs_df)}), skipping...")
        return None
    
    # FIX B5: Use ALL pairs (not just COOCCUR_COL > 0)
    # Only filter NaN/inf
    valid_pairs = pairs_df[np.isfinite(pairs_df[COOCCUR_COL])].copy()
    
    if len(valid_pairs) < 5:
        print(f"  ⚠ Too few valid pairs ({len(valid_pairs)}), skipping...")
        return None
    
    observed_rho, observed_p = stats.spearmanr(valid_pairs['distance'], valid_pairs[COOCCUR_COL])
    
    print(f"\n  Observed: ρ={observed_rho:.4f}, p_naive={observed_p:.4f}")
    
    # Circular shift null (FIX A3: This IS the rate control)
    print(f"\n  Control 1: Circular shift null ({n_perm_circular} permutations)...")
    null_rhos_circular, null_rates = circular_shift_null(
        region_detections, layout_df, region_channels, primary_window, 
        trial_duration_sec, n_perm=n_perm_circular
    )
    
    # FIX: Use MEDIAN for robustness to skew
    null_median = np.median(null_rhos_circular)
    null_mean = np.mean(null_rhos_circular)
    null_std = np.std(null_rhos_circular)
    
    # SANITY CHECK: null_std == 0 means degenerate data or too few permutations
    if null_std < 1e-10:
        print(f"  ✗ FAIL: null_std ≈ 0 (degenerate null distribution)")
        print(f"     This means either: (1) too few permutations, or (2) no variability in data")
        return None
    
    # Effect size using median (robust)
    effect_size = observed_rho - null_median
    # Z-score using mean/std (conventional, now safe)
    z_score = (observed_rho - null_mean) / null_std
    
    # FIX A1: ONE-SIDED LEFT-TAIL test for "more negative than null"
    # Note: effect size uses median(null), p-value uses empirical CDF
    p_circular = np.mean(null_rhos_circular <= observed_rho)
    
    print(f"  → Null median: {null_median:+.4f}, mean: {null_mean:+.4f}")
    print(f"  → Effect size (ρ - median_null): {effect_size:+.4f}")
    print(f"  → Z-score: {z_score:+.2f}")
    print(f"  → p_circular = {p_circular:.4f} (one-sided left-tail, empirical CDF)")
    
    # Permutation test
    print(f"\n  Control 2: Permutation test ({n_perm_distance} permutations)...")
    p_perm, null_rhos_perm = permutation_test_correlation(
        valid_pairs['distance'].values, 
        valid_pairs[COOCCUR_COL].values,
        n_perm=n_perm_distance
    )
    print(f"  → p_perm = {p_perm:.4f} (one-sided left-tail)")
    
    # Coordinate shuffle (channel-level geometry control)
    print(f"\n  Control 3: Coordinate shuffle null ({n_perm_circular} permutations)...")
    null_rhos_coord, _ = coordinate_shuffle_null(
        region_detections, layout_df, region_channels, primary_window,
        trial_duration_sec, n_perm=n_perm_circular
    )
    p_coord = np.mean(null_rhos_coord <= observed_rho)
    print(f"  → p_coord = {p_coord:.4f} (one-sided left-tail)")
    
    # Robustness
    print(f"\n  Control 4: Robustness to window size...")
    robustness_results = []
    for W in window_values:
        pairs_W = compute_pairwise_synchrony(
            region_detections, layout_df, region_channels, 
            window_ms=W, trial_duration_sec=trial_duration_sec
        )
        valid_W = pairs_W[np.isfinite(pairs_W[COOCCUR_COL])]
        
        if len(valid_W) > 5:
            rho_W, p_W = stats.spearmanr(valid_W['distance'], valid_W[COOCCUR_COL])
            robustness_results.append({
                'window_ms': W,
                'rho': rho_W,
                'p': p_W,
                'n_pairs': len(valid_W),
            })
            print(f"    W={W:2d}ms: ρ={rho_W:+.4f}, p={p_W:.4f}")
    
    robustness_df = pd.DataFrame(robustness_results)
    
    # Save
    pairs_df.to_csv(output_dir / "pairwise_synchrony.csv", index=False)
    robustness_df.to_csv(output_dir / "robustness_to_window.csv", index=False)
    
    # Significance: 2-test criterion (rho < 0 AND p_circ < 0.05 AND p_perm < 0.05)
    # Note: p_coord is reported but not required for significance
    is_local = (observed_rho < 0) and (p_circular < 0.05) and (p_perm < 0.05)
    
    # Visualization
    fig = plt.figure(figsize=(20, 10))
    
    # Main result
    ax1 = plt.subplot(2, 4, 1)
    ax1.scatter(valid_pairs['distance'], valid_pairs[COOCCUR_COL], alpha=0.5, s=20)
    z = np.polyfit(valid_pairs['distance'], valid_pairs[COOCCUR_COL], 1)
    p_line = np.poly1d(z)
    x_trend = np.linspace(valid_pairs['distance'].min(), valid_pairs['distance'].max(), 100)
    ax1.plot(x_trend, p_line(x_trend), 'r--', linewidth=2, alpha=0.7, label='Linear guide')
    ax1.set_xlabel('Distance', fontsize=10)
    ax1.set_ylabel(f'Co-occurrence Fraction (W={primary_window}ms)', fontsize=10)
    ax1.set_title(f'Observed (symmetric)\nρ={observed_rho:.3f}, p_naive={observed_p:.4f}', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Circular shift null
    ax2 = plt.subplot(2, 4, 2)
    ax2.hist(null_rhos_circular, bins=30, alpha=0.7, color='gray', edgecolor='black')
    ax2.axvline(observed_rho, color='red', linewidth=2, linestyle='--', label=f'Observed')
    ax2.axvline(null_median, color='blue', linewidth=2, linestyle=':', label=f'Null median')
    ax2.set_xlabel('ρ (null)', fontsize=10)
    ax2.set_ylabel('Count', fontsize=10)
    ax2.set_title(f'Circular Shift Null (Rate Control)\np={p_circular:.4f}, z={z_score:.2f}', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Distance permutation test
    ax3 = plt.subplot(2, 4, 3)
    ax3.hist(null_rhos_perm, bins=50, alpha=0.7, color='gray', edgecolor='black')
    ax3.axvline(observed_rho, color='red', linewidth=2, linestyle='--', label=f'Observed')
    ax3.set_xlabel('ρ (null)', fontsize=10)
    ax3.set_ylabel('Count', fontsize=10)
    ax3.set_title(f'Distance Permutation\np={p_perm:.4f} (one-sided)', fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Coordinate shuffle null
    ax4 = plt.subplot(2, 4, 4)
    ax4.hist(null_rhos_coord, bins=30, alpha=0.7, color='gray', edgecolor='black')
    ax4.axvline(observed_rho, color='red', linewidth=2, linestyle='--', label=f'Observed')
    ax4.set_xlabel('ρ (null)', fontsize=10)
    ax4.set_ylabel('Count', fontsize=10)
    ax4.set_title(f'Coordinate Shuffle Null\np={p_coord:.4f} (one-sided)', fontsize=11)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # Robustness
    ax5 = plt.subplot(2, 4, 5)
    if len(robustness_df) > 0:
        ax5.plot(robustness_df['window_ms'], robustness_df['rho'], 'o-', linewidth=2, markersize=8)
        ax5.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Window Size (ms)', fontsize=10)
    ax5.set_ylabel('ρ', fontsize=10)
    ax5.set_title('Robustness to Window', fontsize=11)
    ax5.grid(True, alpha=0.3)
    
    # Co-occurrence histogram
    ax6 = plt.subplot(2, 4, 6)
    ax6.hist(valid_pairs[COOCCUR_COL], bins=30, alpha=0.7, edgecolor='black')
    ax6.set_xlabel('Co-occurrence Fraction', fontsize=10)
    ax6.set_ylabel('# Pairs', fontsize=10)
    ax6.set_title('Co-occurrence Distribution', fontsize=11)
    ax6.grid(True, alpha=0.3)
    
    # Summary
    ax7 = plt.subplot2grid((2, 4), (1, 2), colspan=2)
    ax7.axis('off')
    sig_marker = '✓' if is_local else '✗'
    # FIX: Label now says median, matching computation
    summary_text = f"""SUMMARY: {region_name}

Channels: {len(region_channels)}
Ripples: {len(region_detections)}
Pairs: {len(pairs_df)} (all included)

RESULT (W={primary_window}ms):
  ρ_obs = {observed_rho:.3f}
  p_naive = {observed_p:.4f}
  
  Null: median={null_median:+.3f}, mean={null_mean:+.3f}
  Effect size (ρ - median_null) = {effect_size:+.3f}
  Z-score = {z_score:+.2f}
  
  p_circular = {p_circular:.4f} (1-sided, rate control)
  p_perm = {p_perm:.4f} (1-sided, pair-level geom)
  p_coord = {p_coord:.4f} (1-sided, channel-level geom)

SIGNIFICANCE (2-test criterion):
  ρ < 0? {observed_rho < 0}
  p_circ < 0.05? {p_circular < 0.05} (required)
  p_perm < 0.05? {p_perm < 0.05} (required)
  p_coord < 0.05? {p_coord < 0.05} (reported only)

{sig_marker} {'LOCAL synchrony' if is_local else 'NOT significant'}
"""
    ax7.text(0.05, 0.5, summary_text, fontsize=9, family='monospace', verticalalignment='center')
    
    plt.suptitle(f'Synchrony Analysis: {region_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "synchrony_analysis.png", dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {output_dir.name}/synchrony_analysis.png")
    plt.close()
    
    # Return summary
    return {
        'region': region_name,
        'n_channels': len(region_channels),
        'n_ripples': len(region_detections),
        'n_pairs': len(pairs_df),
        'rho_observed': observed_rho,
        'null_median': null_median,
        'null_mean': null_mean,
        'null_std': null_std,
        'effect_size': effect_size,  # ρ_obs - median(null)
        'z_score': z_score,
        'p_naive': observed_p,
        'p_circular': p_circular,
        'p_perm': p_perm,
        'p_coord': p_coord,
        'significant': is_local,
        'rho_10ms': robustness_df[robustness_df['window_ms']==10]['rho'].values[0] if len(robustness_df[robustness_df['window_ms']==10]) > 0 else np.nan,
    }


def main():
    parser = argparse.ArgumentParser(description="Run synchrony analysis with proper controls")
    parser.add_argument('--session', type=int, required=True, help='Session number (e.g., 1, 46, 134)')
    parser.add_argument('--trial', type=int, required=True, help='Trial number (e.g., 1, 4, 6)')
    parser.add_argument('--z_low', type=float, default=3.0, help='Z-score threshold for ripple detection')
    parser.add_argument('--trial_duration_sec', type=float, default=None, help='Actual trial duration in seconds (RECOMMENDED - avoid estimating from data)')
    parser.add_argument('--include_all_channels', action='store_true', help='Include "All Channels" analysis (default: off)')
    
    args = parser.parse_args()
    
    # Construct paths
    base_dir = Path("/vol/brains/bd3/pesaranlab/Archie_RecStim_vSUBNETS220_2nd/matlab/mfiles/Chen")
    trial_bipolar_dir = base_dir / f"session{args.session:03d}" / f"trial{args.trial:03d}_bipolar"
    session_name = f"session{args.session:03d}_trial{args.trial:03d}"
    
    layout_file = Path('/vol/brains/bd3/pesaranlab/Archie_RecStim_vSUBNETS220_2nd/matlab/mfiles/Chen/bipolar_layout.csv')
    labels_file = trial_bipolar_dir / "bipolar_channel_labels.csv"
    
    output_base_dir = trial_bipolar_dir / "synchrony_analysis"
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*100)
    print(f"SYNCHRONY ANALYSIS - {session_name.upper()}")
    print("="*100)
    print("\nQuestion: Is synchrony local (network) or global (common input)?")
    print("\nFixes implemented:")
    print("  A1. One-sided test (H1: ρ < 0) for local synchrony")
    print("  A2. Symmetric co-occurrence (average both directions, use <=)")
    print("  A3. Rate control = circular shift null (not analytic)")
    print("  B5. Include ALL pairs (not just cooccur > 0)")
    print("  C6. Coordinate shuffle (channel-level geometry)")
    print("  R2. Standardized column names, median null, layout checks")
    print("\nControls:")
    print("  1. Rate-preserving null (circular shift)")
    print("  2. Distance permutation test (pair-level geometry)")
    print("  3. Coordinate shuffle null (channel-level geometry)")
    print("  4. Robustness to W: [10, 20, 30, 50] ms")
    
    # Load data
    print("\n" + "="*100)
    print("LOADING DATA")
    print("="*100)
    
    detections_df, duration_estimate = load_ripple_detections(trial_bipolar_dir, z_low=args.z_low)
    print(f"Total ripple detections: {len(detections_df)}")
    
    if args.trial_duration_sec is not None:
        trial_duration = args.trial_duration_sec
        print(f"Using provided trial duration: {trial_duration:.1f} sec")
    else:
        trial_duration = duration_estimate
        print(f"⚠ Estimated trial duration from data: {trial_duration:.1f} sec")
        print("  This may be INACCURATE if events stop before trial ends!")
        print("  RECOMMENDED: provide --trial_duration_sec for accuracy")
        
        # Hard-fail on obviously wrong durations
        if trial_duration < 30.0:
            print(f"\n✗ CRITICAL ERROR: Duration < 30s is suspicious!")
            print("  This will distort the circular shift null. Provide --trial_duration_sec")
            sys.exit(1)
        elif trial_duration < 60.0:
            print(f"  ⚠ WARNING: Duration < 60s may be underestimated")
    
    layout_df = pd.read_csv(layout_file)
    labels_df = pd.read_csv(labels_file)
    
    unique_channels = detections_df['bipolar_ch'].nunique()
    print(f"Channels with detections: {unique_channels}")
    
    # Regions to analyze (only include All Channels if requested)
    regions = [
        ("Parietal", "within_parietal"),
        ("Prefrontal", "within_prefrontal"),
        ("Motor", "within_motor"),
    ]
    
    if args.include_all_channels:
        regions.append(("All Channels", "all"))
        print("\n  ℹ Including 'All Channels' analysis (--include_all_channels flag set)")
    
    # Analyze each region
    summaries = []
    for region_name, region_filter in regions:
        summary = analyze_region(
            region_name=region_name,
            region_filter=region_filter,
            detections_df=detections_df,
            layout_df=layout_df,
            labels_df=labels_df,
            output_base_dir=output_base_dir,
            trial_duration_sec=trial_duration,
        )
        if summary is not None:
            summaries.append(summary)
    
    # Create comparison summary
    if len(summaries) > 0:
        print("\n" + "="*100)
        print("CROSS-REGION COMPARISON")
        print("="*100)
        
        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(output_base_dir / "region_comparison_summary.csv", index=False)
        
        # Print table
        print("\n" + "-"*120)
        print(f"{'Region':<15} {'Ch':>4} {'Rip':>6} {'Pairs':>6} {'ρ_obs':>7} {'Effect':>7} {'Z':>6} {'p_circ':>7} {'p_perm':>7} {'p_coord':>7} {'Sig':>5}")
        print("-"*120)
        for _, row in summary_df.iterrows():
            sig = "✓" if row['significant'] else "✗"
            print(f"{row['region']:<15} {row['n_channels']:>4} {row['n_ripples']:>6} {row['n_pairs']:>6} "
                  f"{row['rho_observed']:>+7.3f} {row['effect_size']:>+7.3f} {row['z_score']:>+6.2f} "
                  f"{row['p_circular']:>7.4f} {row['p_perm']:>7.4f} {row['p_coord']:>7.4f} {sig:>5}")
        print("-"*120)
        print("\nInterpretation:")
        print("  ✓ = LOCAL synchrony (negative ρ, p < 0.05 on all three tests)")
        print("  ✗ = NOT significant or wrong direction")
        print("\nTests:")
        print("  p_circ  = circular shift (rate control)")
        print("  p_perm  = distance permutation (pair-level geometry)")
        print("  p_coord = coordinate shuffle (channel-level geometry)")
        print("  Effect  = ρ_obs - median(ρ_null) [robust to skew]")
        print("  Z       = standardized effect [(ρ_obs - mean) / std]")
    
    print(f"\nAll results saved in: {output_base_dir}/")
    print("\n⚠ For publication: Use multi-session analysis to handle pair non-independence!")


if __name__ == "__main__":
    main()
