#!/usr/bin/env python3
"""
Compare ripple rates between two trials (e.g., pre-stim vs post-stim).

Computes:
- Overall ripple rate (ripples/sec) per channel
- Region-specific rates (parietal, prefrontal, motor)
- Statistical comparison (paired t-test, Wilcoxon)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
import matplotlib.pyplot as plt
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'packages' / 'ripple_core'))


def load_ripple_counts(trial_dir: Path, z_low: float = 2.5):
    """Load ripple counts per channel."""
    counts = {}
    
    for ch_dir in sorted(trial_dir.glob("b[0-9][0-9][0-9]")):
        if not ch_dir.is_dir():
            continue
        channel = ch_dir.name
        
        # Find ripple file
        for pattern in [f"ripples_{channel}_zlow{z_low}.mat", 
                        f"ripples_{channel}_zlow2.5.mat"]:
            mat_file = ch_dir / pattern
            if mat_file.exists():
                try:
                    data = sio.loadmat(str(mat_file), squeeze_me=True)
                    starts = None
                    for key in ['merged_starts', 'starts']:
                        if key in data and data[key] is not None:
                            starts = np.atleast_1d(data[key])
                            break
                    counts[channel] = len(starts) if starts is not None else 0
                except:
                    counts[channel] = 0
                break
    
    return counts


def load_channel_labels(trial_dir: Path):
    """Load channel region labels."""
    labels_file = trial_dir / "bipolar_channel_labels.csv"
    if labels_file.exists():
        return pd.read_csv(labels_file)
    return None


def get_trial_duration(trial_dir: Path):
    """Estimate trial duration from first channel's data."""
    # Try to find any .mat file with LFP data
    for ch_dir in sorted(trial_dir.parent.glob("trial[0-9][0-9][0-9]")):
        if ch_dir.name == trial_dir.name.replace("_bipolar", ""):
            # Found raw trial dir
            for chan_dir in sorted(ch_dir.glob("chan*")):
                for mat_file in chan_dir.glob("*.mat"):
                    try:
                        data = sio.loadmat(str(mat_file), squeeze_me=True)
                        if 'lfp' in data:
                            fs = data.get('fs', 1000)
                            return len(data['lfp']) / fs
                    except:
                        continue
    
    # Fallback: estimate from ripple times
    for ch_dir in sorted(trial_dir.glob("b[0-9][0-9][0-9]")):
        for mat_file in ch_dir.glob("ripples_*.mat"):
            try:
                data = sio.loadmat(str(mat_file), squeeze_me=True)
                fs = data.get('fs', 1000)
                for key in ['merged_ends', 'ends']:
                    if key in data and data[key] is not None:
                        ends = np.atleast_1d(data[key])
                        return ends.max() / fs
            except:
                continue
    
    return 300.0  # Default 5 minutes


def compare_trials(session: int, trial1: int, trial2: int, 
                   label1: str = "Trial 1", label2: str = "Trial 2",
                   z_low: float = 2.5):
    """Compare ripple rates between two trials."""
    
    base = Path("/vol/brains/bd3/pesaranlab/Archie_RecStim_vSUBNETS220_2nd/matlab/mfiles/Chen")
    
    trial1_dir = base / f"session{session:03d}" / f"trial{trial1:03d}_bipolar"
    trial2_dir = base / f"session{session:03d}" / f"trial{trial2:03d}_bipolar"
    
    print("=" * 80)
    print(f"RIPPLE RATE COMPARISON: Session {session}")
    print(f"  {label1}: Trial {trial1:03d}")
    print(f"  {label2}: Trial {trial2:03d}")
    print("=" * 80)
    
    # Load counts
    counts1 = load_ripple_counts(trial1_dir, z_low)
    counts2 = load_ripple_counts(trial2_dir, z_low)
    
    # Get durations
    dur1 = get_trial_duration(trial1_dir)
    dur2 = get_trial_duration(trial2_dir)
    
    print(f"\nDurations: {label1}={dur1:.1f}s, {label2}={dur2:.1f}s")
    
    # Common channels
    common_channels = sorted(set(counts1.keys()) & set(counts2.keys()))
    print(f"Common channels: {len(common_channels)}")
    
    # Compute rates
    rates1 = {ch: counts1[ch] / dur1 for ch in common_channels}
    rates2 = {ch: counts2[ch] / dur2 for ch in common_channels}
    
    # Load labels for region analysis
    labels_df = load_channel_labels(trial1_dir)
    
    # Create comparison DataFrame
    comparison = []
    for ch in common_channels:
        region = "unknown"
        if labels_df is not None:
            row = labels_df[labels_df['bipolar_channel'] == ch]
            if len(row) > 0:
                region = row.iloc[0]['region_type']
        
        comparison.append({
            'channel': ch,
            'region': region,
            f'count_{label1}': counts1[ch],
            f'count_{label2}': counts2[ch],
            f'rate_{label1}': rates1[ch],
            f'rate_{label2}': rates2[ch],
            'rate_change': rates2[ch] - rates1[ch],
            'rate_ratio': rates2[ch] / rates1[ch] if rates1[ch] > 0 else np.nan,
            'pct_change': 100 * (rates2[ch] - rates1[ch]) / rates1[ch] if rates1[ch] > 0 else np.nan
        })
    
    df = pd.DataFrame(comparison)
    
    # Overall statistics
    print("\n" + "=" * 80)
    print("OVERALL COMPARISON")
    print("=" * 80)
    
    r1_arr = np.array([rates1[ch] for ch in common_channels])
    r2_arr = np.array([rates2[ch] for ch in common_channels])
    
    print(f"\n{'Metric':<25} {label1:<15} {label2:<15} {'Change':<15}")
    print("-" * 70)
    print(f"{'Total ripples':<25} {sum(counts1.values()):<15} {sum(counts2.values()):<15} "
          f"{sum(counts2.values()) - sum(counts1.values()):+}")
    print(f"{'Mean rate (rip/s)':<25} {np.mean(r1_arr):.3f}{'':<10} {np.mean(r2_arr):.3f}{'':<10} "
          f"{np.mean(r2_arr) - np.mean(r1_arr):+.3f}")
    print(f"{'Median rate (rip/s)':<25} {np.median(r1_arr):.3f}{'':<10} {np.median(r2_arr):.3f}{'':<10} "
          f"{np.median(r2_arr) - np.median(r1_arr):+.3f}")
    
    # Statistical tests
    t_stat, t_pval = stats.ttest_rel(r1_arr, r2_arr)
    w_stat, w_pval = stats.wilcoxon(r1_arr, r2_arr)
    
    print(f"\n{'Paired t-test:':<25} t={t_stat:.3f}, p={t_pval:.4f}")
    print(f"{'Wilcoxon signed-rank:':<25} W={w_stat:.1f}, p={w_pval:.4f}")
    
    mean_pct_change = df['pct_change'].mean()
    print(f"\n{'Mean % change:':<25} {mean_pct_change:+.1f}%")
    
    # By region
    print("\n" + "=" * 80)
    print("BY REGION")
    print("=" * 80)
    
    region_map = {
        'parietal': 'within_parietal',
        'prefrontal': 'within_prefrontal', 
        'motor': 'within_motor'
    }
    
    print(f"\n{'Region':<15} {'N':<5} {label1:<12} {label2:<12} {'Δ Rate':<12} {'% Change':<12} {'p-value':<10}")
    print("-" * 80)
    
    for region_name, region_key in region_map.items():
        region_df = df[df['region'].str.contains(region_key, case=False, na=False)]
        if len(region_df) > 0:
            r1_region = region_df[f'rate_{label1}'].values
            r2_region = region_df[f'rate_{label2}'].values
            
            if len(r1_region) > 1:
                _, p_region = stats.wilcoxon(r1_region, r2_region)
            else:
                p_region = np.nan
            
            delta = np.mean(r2_region) - np.mean(r1_region)
            pct = 100 * delta / np.mean(r1_region) if np.mean(r1_region) > 0 else 0
            
            print(f"{region_name.capitalize():<15} {len(region_df):<5} "
                  f"{np.mean(r1_region):.3f}{'':<7} {np.mean(r2_region):.3f}{'':<7} "
                  f"{delta:+.3f}{'':<7} {pct:+.1f}%{'':<7} "
                  f"{'p=' + f'{p_region:.4f}' if not np.isnan(p_region) else 'N/A'}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Scatter plot of rates
    ax1 = axes[0, 0]
    ax1.scatter(r1_arr, r2_arr, alpha=0.5, s=30)
    max_rate = max(r1_arr.max(), r2_arr.max()) * 1.1
    ax1.plot([0, max_rate], [0, max_rate], 'k--', alpha=0.5, label='No change')
    ax1.set_xlabel(f'{label1} rate (ripples/s)')
    ax1.set_ylabel(f'{label2} rate (ripples/s)')
    ax1.set_title(f'Channel-by-Channel Rate Comparison\n(p={w_pval:.4f}, Wilcoxon)')
    ax1.legend()
    ax1.set_xlim(0, max_rate)
    ax1.set_ylim(0, max_rate)
    
    # 2. Histogram of rate changes
    ax2 = axes[0, 1]
    changes = r2_arr - r1_arr
    ax2.hist(changes, bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='red', linestyle='--', linewidth=2)
    ax2.axvline(np.mean(changes), color='green', linestyle='-', linewidth=2, 
                label=f'Mean: {np.mean(changes):+.3f}')
    ax2.set_xlabel('Rate change (ripples/s)')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Rate Changes')
    ax2.legend()
    
    # 3. By region boxplot
    ax3 = axes[1, 0]
    region_data = []
    region_labels = []
    for region_name, region_key in region_map.items():
        region_df = df[df['region'].str.contains(region_key, case=False, na=False)]
        if len(region_df) > 0:
            region_data.append(region_df['pct_change'].values)
            region_labels.append(region_name.capitalize())
    
    if region_data:
        bp = ax3.boxplot(region_data, labels=region_labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax3.axhline(0, color='red', linestyle='--')
        ax3.set_ylabel('% Change in Rate')
        ax3.set_title('Rate Change by Region')
    
    # 4. Paired comparison
    ax4 = axes[1, 1]
    x = np.arange(len(common_channels))
    width = 0.35
    ax4.bar(x - width/2, r1_arr, width, label=label1, alpha=0.7)
    ax4.bar(x + width/2, r2_arr, width, label=label2, alpha=0.7)
    ax4.set_xlabel('Channel')
    ax4.set_ylabel('Rate (ripples/s)')
    ax4.set_title('Rate by Channel')
    ax4.legend()
    ax4.set_xticks([])  # Too many channels to label
    
    plt.suptitle(f'Session {session}: {label1} vs {label2}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_dir = base / f"session{session:03d}" / "rate_comparison"
    output_dir.mkdir(exist_ok=True)
    
    fig.savefig(output_dir / f"rate_comparison_trial{trial1:03d}_vs_trial{trial2:03d}.png", 
                dpi=150, bbox_inches='tight')
    df.to_csv(output_dir / f"rate_comparison_trial{trial1:03d}_vs_trial{trial2:03d}.csv", 
              index=False)
    
    print(f"\n\nSaved to: {output_dir}/")
    plt.close()
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Compare ripple rates between trials')
    parser.add_argument('--session', type=int, required=True)
    parser.add_argument('--trial1', type=int, required=True, help='First trial (e.g., pre-stim)')
    parser.add_argument('--trial2', type=int, required=True, help='Second trial (e.g., post-stim)')
    parser.add_argument('--label1', type=str, default='Pre-Stim')
    parser.add_argument('--label2', type=str, default='Post-Stim')
    parser.add_argument('--z_low', type=float, default=2.5)
    args = parser.parse_args()
    
    compare_trials(args.session, args.trial1, args.trial2, 
                   args.label1, args.label2, args.z_low)


if __name__ == "__main__":
    main()
