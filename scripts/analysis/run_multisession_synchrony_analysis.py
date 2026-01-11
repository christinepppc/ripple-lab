#!/usr/bin/env python3
"""
Multi-Session Synchrony Analysis - Publication Quality Statistics

CRITICAL FIX: Addresses pair non-independence issue.

Problem: ~435 pairs share channels → pairs aren't independent.
         Naive statistics treat pairs as i.i.d., which is WRONG.

Solution: Make the unit of analysis = session/trial.

Per Session:
  1. Compute ρ_obs
  2. Compute null distribution (circular shifts) → ρ_null^k
  3. Compute session-level effect: ρ_obs - median(ρ_null) [robust to skew]
  4. Compute z-score: (ρ_obs - mean(ρ_null)) / std(ρ_null)

Across Sessions:
  PRIMARY TEST (determines significance):
    - Wilcoxon signed-rank test: H1: median(effect) < 0
  
  SECONDARY TESTS (supporting evidence, NOT independent):
    - Fisher's method: combine p-values (direction-agnostic)
    - Stouffer's method: combine z-scores (directional)

FIXES (Round 2):
  - Stouffer: correct one-sided p-value conversion
  - Test hierarchy: Wilcoxon primary, Fisher/Stouffer secondary
  - Effect size: uses median(null) for robustness
  - Added Wilcoxon zero_method parameter
  - Layout consistency checks and warnings

Usage:
    # 1. Create sessions list
    cat > sessions.csv << EOF
    session_name,trial_dir,trial_duration_sec
    session134_trial001,/path/to/trial001_bipolar,300
    session135_trial001,/path/to/trial002_bipolar,300
    EOF

    # 2. Run
    python scripts/analysis/run_multisession_synchrony_analysis.py \
        --sessions_file sessions.csv \
        --output_dir /path/to/output \
        --z_low 3.0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages"))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import from single-session analysis
from run_synchrony_analysis import (
    load_ripple_detections,
    compute_pairwise_synchrony,
    circular_shift_null,
    coordinate_shuffle_null,
    permutation_test_correlation,
    COOCCUR_COL,  # CANONICAL COLUMN NAME
)


def combine_sessions_fisher(p_values: np.ndarray) -> Tuple[float, float]:
    """
    Combine p-values across sessions using Fisher's method.
    
    Fisher's method is direction-agnostic and requires two-sided p-values.
    We convert one-sided p-values to two-sided before combining.
    
    Returns (chi2_stat, combined_p_value)
    """
    # Convert one-sided to two-sided: p_2sided = 2 * min(p, 1-p)
    # For left-tail p where small p means negative effect:
    p_twosided = 2 * np.minimum(p_values, 1 - p_values)
    p_twosided = np.clip(p_twosided, 1e-10, 1.0)  # Keep in valid range
    
    # Fisher's method: -2 * sum(log(p_i)) ~ chi^2(2k)
    chi2_stat = -2 * np.sum(np.log(p_twosided))
    df = 2 * len(p_twosided)
    combined_p = 1 - stats.chi2.cdf(chi2_stat, df)
    return chi2_stat, combined_p


def combine_sessions_stouffer(p_values: np.ndarray, weights: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """
    Combine p-values across sessions using Stouffer's method (with optional weights).
    
    For one-sided LEFT-TAIL p-values where small p indicates negative effect:
      p_i = P(null <= obs)
    
    We convert to z-scores (negative for small p), combine with weights, and convert back.
    
    Weighted Stouffer: Z = (Σ w_i * Z_i) / sqrt(Σ w_i^2)
    
    Returns (combined_z, combined_p_value)
    """
    # Convert one-sided p-values to z-scores
    # p_i = P(null <= obs), so for negative effects p is small
    # z_i = Φ^(-1)(p_i) will be negative when p < 0.5
    z_scores = stats.norm.ppf(np.clip(p_values, 1e-10, 1 - 1e-10))
    
    if weights is None:
        # Unweighted: sum(Z_i) / sqrt(k) ~ N(0,1)
        combined_z = np.sum(z_scores) / np.sqrt(len(z_scores))
    else:
        # Weighted: Σ(w_i * Z_i) / sqrt(Σ w_i^2)
        combined_z = np.sum(weights * z_scores) / np.sqrt(np.sum(weights**2))
    
    # Convert combined z back to one-sided p-value
    # H1: combined_z < 0 (negative effect), so left-tail
    combined_p = stats.norm.cdf(combined_z)
    
    return combined_z, combined_p


def analyze_single_session(
    session_name: str,
    trial_bipolar_dir: Path,
    layout_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    region_name: str,
    region_filter: str,
    z_low: float = 3.0,
    trial_duration_sec: Optional[float] = None,
    primary_window: float = 50.0,
    n_perm_circular: int = 500,
    n_perm_distance: int = 5000
) -> Optional[Dict]:
    """
    Analyze a single session and return summary statistics.
    
    Returns dict with:
        - session_name
        - rho_observed
        - null_rhos_circular (array)
        - effect_size: rho_obs - median(null_rhos) [robust]
        - z_score: (rho_obs - mean) / std [standardized]
        - p_circular, p_perm, p_coord (all ONE-SIDED LEFT-TAIL)
        - n_channels, n_ripples, n_pairs
    """
    print(f"\n{'='*80}")
    print(f"SESSION: {session_name} | REGION: {region_name}")
    print(f"{'='*80}")
    
    # Load ripple detections
    detections_df, duration_estimate = load_ripple_detections(trial_bipolar_dir, z_low=z_low)
    
    if trial_duration_sec is not None:
        trial_duration = trial_duration_sec
    else:
        # CRITICAL: Duration estimate is dangerous - hard-fail on suspicious values
        trial_duration = duration_estimate
        print(f"  ⚠ Using estimated duration: {trial_duration:.1f}s (INACCURATE - will distort null!)")
        
        # Hard-fail on obviously wrong durations
        if trial_duration < 30.0:
            print(f"  ✗ FAIL: Duration < 30s is suspicious. Provide --trial_duration_sec!")
            return None
        elif trial_duration < 60.0:
            print(f"  ⚠ WARNING: Duration < 60s may be underestimated.")
    
    # Filter channels by region
    if region_filter == "all":
        region_channels = labels_df['bipolar_channel'].tolist()
    else:
        region_channels = labels_df[labels_df['region_type'] == region_filter]['bipolar_channel'].tolist()
    
    if len(region_channels) < 3:
        print(f"  ⚠ Too few channels ({len(region_channels)}), skipping session...")
        return None
    
    # CHECK: Verify channels in layout
    missing_in_layout = set(region_channels) - set(layout_df['bipolar_ch'])
    if missing_in_layout:
        print(f"  ⚠ WARNING: {len(missing_in_layout)} channels missing in layout!")
        print(f"     First 5: {sorted(list(missing_in_layout))[:5]}")
        region_channels = [ch for ch in region_channels if ch in layout_df['bipolar_ch'].values]
        if len(region_channels) < 3:
            print(f"  ⚠ Too few valid channels, skipping...")
            return None
    
    region_detections = detections_df[detections_df['bipolar_ch'].isin(region_channels)]
    
    if len(region_detections) < 50:
        print(f"  ⚠ Too few detections ({len(region_detections)}), skipping session...")
        return None
    
    # Compute pairwise synchrony
    pairs_df = compute_pairwise_synchrony(
        region_detections, layout_df, region_channels, 
        window_ms=primary_window, 
        trial_duration_sec=trial_duration
    )
    
    if len(pairs_df) < 10:
        print(f"  ⚠ Too few pairs ({len(pairs_df)}), skipping session...")
        return None
    
    # CANONICAL COLUMN NAME (from imported constant)
    valid_pairs = pairs_df[np.isfinite(pairs_df[COOCCUR_COL])].copy()
    
    if len(valid_pairs) < 5:
        print(f"  ⚠ Too few valid pairs ({len(valid_pairs)}), skipping session...")
        return None
    
    # Observed correlation
    observed_rho, _ = stats.spearmanr(valid_pairs['distance'], valid_pairs[COOCCUR_COL])
    print(f"  Observed: ρ = {observed_rho:+.4f}")
    
    # Circular shift null (THIS IS KEY: we get the full distribution per session)
    print(f"  Circular shift null ({n_perm_circular} permutations)...")
    null_rhos_circular, _ = circular_shift_null(
        region_detections, layout_df, region_channels, primary_window, 
        trial_duration, n_perm=n_perm_circular
    )
    
    # Session-level effect size (THIS IS WHAT WE TEST ACROSS SESSIONS)
    # Use median for robustness (nulls can be skewed)
    null_median = np.median(null_rhos_circular)
    null_mean = np.mean(null_rhos_circular)
    null_std = np.std(null_rhos_circular)
    
    # SANITY CHECK: null_std == 0 means degenerate data or too few permutations
    if null_std < 1e-10:
        print(f"  ✗ FAIL: null_std ≈ 0 (degenerate null distribution)")
        print(f"     This means either: (1) too few permutations, or (2) no variability in data")
        return None
    
    effect_size = observed_rho - null_median  # Robust to skew
    z_score = (observed_rho - null_mean) / null_std  # Standardized effect (now safe)
    
    # ONE-SIDED LEFT-TAIL p-value: P(null <= obs)
    # Note: effect size uses median(null), p-value uses empirical CDF
    p_circular = np.mean(null_rhos_circular <= observed_rho)
    
    print(f"  → Null median: {null_median:+.4f}, mean: {null_mean:+.4f}")
    print(f"  → Effect size (ρ - median_null): {effect_size:+.4f}")
    print(f"  → Z-score: {z_score:+.4f}")
    print(f"  → p_circular: {p_circular:.4f} (one-sided left-tail)")
    
    # Permutation test (pair-level geometry) - also ONE-SIDED LEFT-TAIL
    print(f"  Distance permutation test ({n_perm_distance} permutations)...")
    p_perm, _ = permutation_test_correlation(
        valid_pairs['distance'].values, 
        valid_pairs[COOCCUR_COL].values,
        n_perm=n_perm_distance
    )
    print(f"  → p_perm: {p_perm:.4f} (one-sided left-tail)")
    
    # Coordinate shuffle (channel-level geometry) - also ONE-SIDED LEFT-TAIL
    print(f"  Coordinate shuffle null ({n_perm_circular} permutations)...")
    null_rhos_coord, _ = coordinate_shuffle_null(
        region_detections, layout_df, region_channels, primary_window,
        trial_duration, n_perm=n_perm_circular
    )
    p_coord = np.mean(null_rhos_coord <= observed_rho)
    print(f"  → p_coord: {p_coord:.4f} (one-sided left-tail)")
    
    return {
        'session_name': session_name,
        'region': region_name,
        'n_channels': len(region_channels),
        'n_ripples': len(region_detections),
        'n_pairs': len(pairs_df),
        'rho_observed': observed_rho,
        'null_median': null_median,
        'null_mean': null_mean,
        'null_std': null_std,
        'effect_size': effect_size,  # ρ_obs - median(null) for robustness
        'z_score': z_score,  # Standardized effect
        'p_circular': p_circular,  # ONE-SIDED LEFT-TAIL
        'p_perm': p_perm,  # ONE-SIDED LEFT-TAIL
        'p_coord': p_coord,  # ONE-SIDED LEFT-TAIL
        'null_rhos_circular': null_rhos_circular,  # Store for meta-analysis
    }


def analyze_multisession(
    sessions_df: pd.DataFrame,
    layout_df: pd.DataFrame,
    region_name: str,
    region_filter: str,
    output_dir: Path,
    z_low: float = 3.0,
    primary_window: float = 50.0,
) -> Optional[Dict]:
    """
    Analyze multiple sessions for a given region.
    
    Implements proper session-level inference to handle pair non-independence.
    
    ⚠ CRITICAL: layout_df must have CONSISTENT channel names across all sessions.
    If channel naming differs between sessions (e.g., b001 in sess1 = different 
    electrode than b001 in sess2), distances will be WRONG and results garbage!
    """
    print("\n" + "="*100)
    print(f"MULTI-SESSION ANALYSIS: {region_name.upper()}")
    print("="*100)
    
    session_results = []
    
    for _, row in sessions_df.iterrows():
        session_name = row['session_name']
        trial_dir = Path(row['trial_dir'])
        trial_duration = row.get('trial_duration_sec', None)
        
        # Load labels for this session
        labels_file = trial_dir / "bipolar_channel_labels.csv"
        if not labels_file.exists():
            print(f"⚠ Labels not found for {session_name}, skipping...")
            continue
        
        labels_df = pd.read_csv(labels_file)
        
        # Analyze this session
        result = analyze_single_session(
            session_name=session_name,
            trial_bipolar_dir=trial_dir,
            layout_df=layout_df,
            labels_df=labels_df,
            region_name=region_name,
            region_filter=region_filter,
            z_low=z_low,
            trial_duration_sec=trial_duration,
            primary_window=primary_window,
        )
        
        if result is not None:
            session_results.append(result)
    
    if len(session_results) == 0:
        print(f"\n⚠ No valid sessions for {region_name}")
        return None
    
    print(f"\n{'='*100}")
    print(f"CROSS-SESSION INFERENCE: {region_name}")
    print(f"{'='*100}")
    print(f"\nSessions analyzed: {len(session_results)}")
    
    # Extract effect sizes
    effect_sizes = np.array([r['effect_size'] for r in session_results])
    rho_observed = np.array([r['rho_observed'] for r in session_results])
    
    # PRIMARY TEST: Wilcoxon signed-rank test on effect sizes
    # This is the most direct test: do sessions consistently show negative effects?
    # H0: median(effect_size) = 0
    # H1: median(effect_size) < 0 (local synchrony)
    if len(effect_sizes) >= 3:
        # FIX: Add zero_method to handle ties/zeros gracefully
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
            effect_sizes, 
            alternative='less',
            zero_method='wilcox'  # Handles zeros consistently
        )
        print(f"\n✓ PRIMARY TEST: Wilcoxon signed-rank (effect sizes)")
        print(f"  Statistic: {wilcoxon_stat}")
        print(f"  p-value (one-sided): {wilcoxon_p:.6f}")
        print(f"  Median effect: {np.median(effect_sizes):+.4f}")
        print(f"  Mean effect: {np.mean(effect_sizes):+.4f} ± {np.std(effect_sizes):.4f}")
    else:
        wilcoxon_stat = np.nan
        wilcoxon_p = np.nan
        print(f"\n⚠ Too few sessions ({len(effect_sizes)}) for Wilcoxon test")
    
    # SECONDARY TESTS: Combine p-values (supporting evidence, NOT independent)
    # Note: These are derived from the same per-session statistics, so not independent tests
    p_circular_values = np.array([r['p_circular'] for r in session_results])
    p_circular_values_clipped = np.clip(p_circular_values, 1e-10, 1.0)
    
    # Fisher's method: tests if p-values are unusually small (direction-agnostic)
    # Uses two-sided p-values (converted inside function)
    fisher_chi2, fisher_p = combine_sessions_fisher(p_circular_values_clipped)
    
    print(f"\n→ SECONDARY: Fisher's method (sensitivity check)")
    print(f"  χ² statistic: {fisher_chi2:.2f}")
    print(f"  Combined p-value: {fisher_p:.6f}")
    print(f"  (Uses two-sided p-values, direction-agnostic)")
    
    # Stouffer's method: combines z-scores with direction and weights
    # Weight by sqrt(n_pairs) to account for varying sample sizes
    n_pairs_array = np.array([r['n_pairs'] for r in session_results])
    weights = np.sqrt(n_pairs_array)  # Weight by sqrt(N) as in standard Stouffer
    stouffer_z, stouffer_p = combine_sessions_stouffer(p_circular_values_clipped, weights=weights)
    
    print(f"\n→ SECONDARY: Stouffer's method (weighted, sensitivity check)")
    print(f"  Combined Z: {stouffer_z:+.4f}")
    print(f"  Combined p-value: {stouffer_p:.6f}")
    print(f"  Weights: sqrt(n_pairs) = {weights.min():.1f}-{weights.max():.1f}")
    print(f"  (Weighted by sample size, directional like Wilcoxon)")
    
    # Summary statistics
    print(f"\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"  Mean ρ_obs: {np.mean(rho_observed):+.4f} ± {np.std(rho_observed):.4f}")
    print(f"  Median ρ_obs: {np.median(rho_observed):+.4f}")
    print(f"  Mean effect: {np.mean(effect_sizes):+.4f} ± {np.std(effect_sizes):.4f}")
    print(f"  Median effect: {np.median(effect_sizes):+.4f}")
    
    # Bootstrap CI for median effect size
    n_bootstrap = 10000
    bootstrap_medians = []
    for _ in range(n_bootstrap):
        resample = np.random.choice(effect_sizes, size=len(effect_sizes), replace=True)
        bootstrap_medians.append(np.median(resample))
    bootstrap_medians = np.array(bootstrap_medians)
    ci_lower = np.percentile(bootstrap_medians, 2.5)
    ci_upper = np.percentile(bootstrap_medians, 97.5)
    print(f"  Median effect 95% CI: [{ci_lower:+.4f}, {ci_upper:+.4f}] (bootstrap)")
    
    # Extract z-scores for additional info
    z_scores_session = np.array([r['z_score'] for r in session_results])
    print(f"  Mean z-score: {np.mean(z_scores_session):+.4f} ± {np.std(z_scores_session):.4f}")
    
    # Note about effect size computation
    print(f"\n  Note: Effect size = ρ_obs - median(ρ_null) [robust to skew]")
    print(f"        P-values use empirical CDF: P(ρ_null <= ρ_obs)")
    
    # Overall significance (PRIMARY TEST: Wilcoxon)
    # Secondary tests (Fisher, Stouffer) are supporting evidence
    is_significant = (
        (np.median(effect_sizes) < 0) and
        (wilcoxon_p < 0.05 if not np.isnan(wilcoxon_p) else False)
    )
    
    # Check if secondary tests also agree (for reporting)
    secondary_support = (fisher_p < 0.05) and (stouffer_p < 0.05)
    
    print(f"\n" + "="*80)
    print("OVERALL RESULT")
    print("="*80)
    print(f"  Primary test (Wilcoxon): {'✓ SIGNIFICANT' if is_significant else '✗ NOT SIGNIFICANT'}")
    print(f"  Secondary tests agree: {'✓ YES' if secondary_support else '✗ NO'}")
    print(f"  ")
    if is_significant and secondary_support:
        print(f"  🎉 STRONG EVIDENCE for local synchrony (all tests agree)")
    elif is_significant:
        print(f"  ⚠ SIGNIFICANT but secondary tests mixed (effect may be weak/variable)")
    else:
        print(f"  ✗ NOT SIGNIFICANT (no evidence for local synchrony)")
    
    # Save session-level results
    session_df = pd.DataFrame([{
        'session_name': r['session_name'],
        'n_channels': r['n_channels'],
        'n_ripples': r['n_ripples'],
        'n_pairs': r['n_pairs'],
        'rho_observed': r['rho_observed'],
        'null_median': r['null_median'],
        'null_mean': r['null_mean'],
        'null_std': r['null_std'],
        'effect_size': r['effect_size'],
        'z_score': r['z_score'],
        'p_circular': r['p_circular'],
        'p_perm': r['p_perm'],
        'p_coord': r['p_coord'],
    } for r in session_results])
    
    session_file = output_dir / f"{region_name.lower().replace(' ', '_')}_sessions.csv"
    session_df.to_csv(session_file, index=False)
    print(f"\nSaved: {session_file.name}")
    
    # Visualization
    fig = plt.figure(figsize=(16, 10))
    
    # Effect sizes per session
    ax1 = plt.subplot(2, 3, 1)
    sessions_names = [r['session_name'] for r in session_results]
    colors = ['green' if e < 0 else 'red' for e in effect_sizes]
    ax1.barh(range(len(effect_sizes)), effect_sizes, color=colors, alpha=0.7)
    ax1.axvline(0, color='black', linestyle='--', linewidth=2)
    ax1.set_yticks(range(len(sessions_names)))
    ax1.set_yticklabels(sessions_names, fontsize=8)
    # FIX: Label now says median, matching computation
    ax1.set_xlabel('Effect Size (ρ_obs - median(ρ_null))', fontsize=10)
    ax1.set_title(f'{region_name}: Effect Sizes per Session', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Distribution of effect sizes
    ax2 = plt.subplot(2, 3, 2)
    # Filter out NaN values for plotting
    effect_sizes_valid = effect_sizes[~np.isnan(effect_sizes)]
    if len(effect_sizes_valid) > 0:
        ax2.hist(effect_sizes_valid, bins=15, alpha=0.7, edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='H0: effect=0')
        ax2.axvline(np.median(effect_sizes_valid), color='green', linestyle='-', linewidth=2, label=f'Median={np.median(effect_sizes_valid):.3f}')
    else:
        ax2.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_xlabel('Effect Size', fontsize=10)
    ax2.set_ylabel('# Sessions', fontsize=10)
    ax2.set_title('Distribution of Effect Sizes', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Observed ρ per session
    ax3 = plt.subplot(2, 3, 3)
    ax3.barh(range(len(rho_observed)), rho_observed, alpha=0.7)
    ax3.axvline(0, color='black', linestyle='--', linewidth=2)
    ax3.set_yticks(range(len(sessions_names)))
    ax3.set_yticklabels(sessions_names, fontsize=8)
    ax3.set_xlabel('ρ_observed', fontsize=10)
    ax3.set_title('Observed Correlations per Session', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # P-values per session
    ax4 = plt.subplot(2, 3, 4)
    ax4.scatter(range(len(p_circular_values)), p_circular_values, s=100, alpha=0.7)
    ax4.axhline(0.05, color='red', linestyle='--', linewidth=2, label='α=0.05')
    ax4.set_yscale('log')
    ax4.set_xticks(range(len(sessions_names)))
    ax4.set_xticklabels(sessions_names, rotation=45, ha='right', fontsize=8)
    ax4.set_ylabel('p_circular (log scale)', fontsize=10)
    ax4.set_title('P-values per Session', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # Combined tests
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    summary_text = f"""CROSS-SESSION INFERENCE
Region: {region_name}
Sessions: {len(session_results)}

EFFECT SIZES:
  Mean: {np.mean(effect_sizes):+.4f} ± {np.std(effect_sizes):.4f}
  Median: {np.median(effect_sizes):+.4f}
  Mean z-score: {np.mean(z_scores_session):+.4f}

PRIMARY TEST:
  Wilcoxon (effect < 0):
    p = {wilcoxon_p:.6f}
    {'✓ SIGNIFICANT' if is_significant else '✗ NOT SIGNIFICANT'}

SECONDARY TESTS (supporting):
  Fisher (combine p):
    χ² = {fisher_chi2:.2f}, p = {fisher_p:.6f}
  
  Stouffer (combine z):
    Z = {stouffer_z:+.4f}, p = {stouffer_p:.6f}

OVERALL: {'🎉 STRONG' if is_significant and secondary_support else '⚠ MIXED' if is_significant else '✗ NOT SIG'}
"""
    ax5.text(0.05, 0.95, summary_text, fontsize=8, family='monospace', 
             verticalalignment='top', transform=ax5.transAxes)
    
    # Session summary table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    table_text = "SESSION SUMMARY\n\n"
    table_text += f"{'Session':<12} {'ρ_obs':>7} {'Eff':>6} {'Z':>6} {'p':>6}\n"
    table_text += "-" * 42 + "\n"
    for r in session_results:
        table_text += f"{r['session_name'][:12]:<12} {r['rho_observed']:>+7.3f} {r['effect_size']:>+6.3f} {r['z_score']:>+6.2f} {r['p_circular']:>6.3f}\n"
    ax6.text(0.05, 0.95, table_text, fontsize=7, family='monospace',
             verticalalignment='top', transform=ax6.transAxes)
    
    plt.suptitle(f'Multi-Session Synchrony Analysis: {region_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_file = output_dir / f"{region_name.lower().replace(' ', '_')}_multisession.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_file.name}")
    plt.close()
    
    return {
        'region': region_name,
        'n_sessions': len(session_results),
        'mean_rho_obs': np.mean(rho_observed),
        'median_rho_obs': np.median(rho_observed),
        'mean_effect': np.mean(effect_sizes),
        'median_effect': np.median(effect_sizes),
        'effect_ci_lower': ci_lower,
        'effect_ci_upper': ci_upper,
        'mean_zscore': np.mean(z_scores_session),
        'wilcoxon_p': wilcoxon_p,
        'fisher_p': fisher_p,
        'stouffer_p': stouffer_p,
        'significant': is_significant,
        'secondary_support': secondary_support,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-session synchrony analysis (fixes pair non-independence)")
    parser.add_argument('--sessions_file', type=str, required=True, help='CSV file with session information')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--z_low', type=float, default=3.0, help='Z-score threshold for ripple detection')
    parser.add_argument('--layout_file', type=str, default='/vol/brains/bd3/pesaranlab/Archie_RecStim_vSUBNETS220_2nd/matlab/mfiles/Chen/bipolar_layout.csv', help='Bipolar layout CSV file')
    
    args = parser.parse_args()
    
    # Load sessions
    sessions_df = pd.read_csv(args.sessions_file)
    print(f"Loaded {len(sessions_df)} sessions from {args.sessions_file}")
    
    # Check for animal_id column
    has_animal_id = 'animal_id' in sessions_df.columns
    if has_animal_id:
        n_animals = sessions_df['animal_id'].nunique()
        print(f"  Animals: {n_animals}")
        print(f"  ⚠ NOTE: If multiple sessions per animal, consider animal-level analysis")
        print(f"           (average within animal, then test across animals)")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load layout
    layout_df = pd.read_csv(args.layout_file)
    
    print("\n" + "="*100)
    print("MULTI-SESSION SYNCHRONY ANALYSIS")
    print("="*100)
    print("\nThis analysis addresses pair non-independence by treating session as the unit of analysis.")
    print("\n⚠ CRITICAL REQUIREMENTS:")
    print("   1. All sessions must use CONSISTENT channel naming!")
    print("      If 'b001' means different electrodes across sessions, results will be WRONG.")
    print("   2. Provide trial_duration_sec for each session (don't rely on estimates)")
    print("   3. If multiple sessions per animal, consider animal-level aggregation")
    print("\nRegions to analyze:")
    
    # Analyze each region
    regions = [
        ("Parietal", "within_parietal"),
        ("Prefrontal", "within_prefrontal"),
        ("Motor", "within_motor"),
    ]
    
    summaries = []
    for region_name, region_filter in regions:
        print(f"  - {region_name}")
        summary = analyze_multisession(
            sessions_df=sessions_df,
            layout_df=layout_df,
            region_name=region_name,
            region_filter=region_filter,
            output_dir=output_dir,
            z_low=args.z_low,
        )
        if summary is not None:
            summaries.append(summary)
    
    # Overall summary
    if len(summaries) > 0:
        print("\n" + "="*100)
        print("OVERALL SUMMARY")
        print("="*100)
        
        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(output_dir / "multisession_summary.csv", index=False)
        
        print("\n" + "-"*130)
        print(f"{'Region':<15} {'N':>3} {'Med_ρ':>7} {'Med_Eff':>8} {'95% CI':>18} {'Z':>6} {'p_Wilc':>8} {'p_Fish':>8} {'p_Stou':>8} {'Sig':>4} {'Supp':>4}")
        print("-"*130)
        for _, row in summary_df.iterrows():
            sig = "✓" if row['significant'] else "✗"
            support = "✓" if row['secondary_support'] else "✗"
            ci_str = f"[{row['effect_ci_lower']:+.3f},{row['effect_ci_upper']:+.3f}]"
            print(f"{row['region']:<15} {row['n_sessions']:>3} {row['median_rho_obs']:>+7.3f} {row['median_effect']:>+8.4f} "
                  f"{ci_str:>18} {row['mean_zscore']:>+6.2f} {row['wilcoxon_p']:>8.5f} {row['fisher_p']:>8.5f} "
                  f"{row['stouffer_p']:>8.5f} {sig:>4} {support:>4}")
        print("-"*130)
        
        print(f"\nAll results saved in: {output_dir}/")
        print("\nInterpretation:")
        print("  Sig:  PRIMARY test (Wilcoxon on effect sizes) - THIS DETERMINES SIGNIFICANCE")
        print("  Supp: SECONDARY tests agree (Fisher & Stouffer) - SUPPORTING EVIDENCE")
        print("  95% CI: Bootstrap confidence interval for median effect size")
        print("  🎉 = Both ✓ = Strong evidence (all tests agree)")
        print("  ⚠ = Sig ✓ but Supp ✗ = Significant but use caution (effect may be weak/variable)")
        print("  ✗ = Not significant")
        print("\nStatistical Notes:")
        print("  - Effect size = ρ_obs - median(ρ_null) [robust to skew]")
        print("  - P-values use empirical CDF [P(ρ_null <= ρ_obs)]")
        print("  - Fisher uses two-sided p-values [converted from one-sided]")
        print("  - Stouffer weighted by sqrt(n_pairs) [accounts for varying sample sizes]")
        print("\nThis multi-session analysis properly handles pair non-independence.")
        print("\nReporting in paper:")
        print("  - Use Wilcoxon p-value as primary significance test")
        print("  - Report median effect with 95% CI")
        print("  - Report Fisher/Stouffer as sensitivity checks (note: not independent of Wilcoxon)")
        print("  - Emphasize that session is the unit of analysis")
        if has_animal_id:
            print("  - If multiple sessions per animal, consider animal-level aggregation")


if __name__ == "__main__":
    main()
