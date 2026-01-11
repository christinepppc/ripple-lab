#!/usr/bin/env python3
"""
Propagation Analysis: Detect traveling waves in ripple events.

For each session:
1. Cluster ripples by time (gap-based)
2. For multi-channel clusters, compute timing delays
3. Test for consistent spatial gradient (plane-fit)
4. Report propagation metrics

Hypothesis: Sessions without local synchrony might have propagation instead!
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add packages to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'packages' / 'ripple_core'))

def load_ripple_events(trial_dir: Path, z_low: float = 2.5):
    """Load ripple events from all channels.
    
    Looks for ripple files in two locations:
    1. trial_dir/b###/ripples_b###_zlow{z_low}.mat (channel subdirectories)
    2. trial_dir/ripple_detections/b###_zlow{z_low}.mat (flat structure)
    """
    events = []
    
    # Method 1: Channel subdirectories (b001/, b002/, etc.)
    for ch_dir in sorted(trial_dir.glob("b[0-9][0-9][0-9]")):
        if not ch_dir.is_dir():
            continue
        channel = ch_dir.name  # e.g., 'b001'
        
        # Try multiple naming patterns (including both z_low values)
        patterns = [
            f"ripples_{channel}_zlow{z_low}.mat",
            f"ripples_{channel}_zlow3.0.mat",  # Fallback
            f"ripples_{channel}_zlow2.5.mat",  # Fallback
            f"ripples_{channel}.mat",
            f"{channel}_ripples_zlow{z_low}.mat"
        ]
        
        for pattern in patterns:
            mat_file = ch_dir / pattern
            if mat_file.exists():
                try:
                    data = sio.loadmat(str(mat_file), squeeze_me=True)
                    
                    # Handle different key names
                    starts = None
                    fs = data.get('fs', 1000.0)  # Sampling frequency
                    for key in ['merged_starts', 'starts', 'start_times', 'ripple_starts']:
                        if key in data and data[key] is not None:
                            starts = np.atleast_1d(data[key])
                            # Convert sample indices to seconds if needed
                            if key == 'merged_starts' and starts.dtype in [np.int32, np.int64]:
                                starts = starts / fs  # Convert samples to seconds
                            break
                    
                    if starts is not None and len(starts) > 0:
                        for start_time in starts:
                            events.append({
                                'channel': channel,
                                'time': start_time * 1000  # Convert to ms
                            })
                except Exception as e:
                    continue
                break
    
    # Method 2: Flat ripple_detections folder
    if not events:
        ripple_dir = trial_dir / "ripple_detections"
        if ripple_dir.exists():
            for mat_file in sorted(ripple_dir.glob(f"*_zlow{z_low}.mat")):
                try:
                    data = sio.loadmat(str(mat_file), squeeze_me=True)
                    channel = mat_file.stem.split('_')[0]
                    
                    starts = None
                    fs = data.get('fs', 1000.0)
                    for key in ['merged_starts', 'starts', 'start_times', 'ripple_starts']:
                        if key in data and data[key] is not None:
                            starts = np.atleast_1d(data[key])
                            if key == 'merged_starts' and starts.dtype in [np.int32, np.int64]:
                                starts = starts / fs
                            break
                    
                    if starts is not None and len(starts) > 0:
                        for start_time in starts:
                            events.append({
                                'channel': channel,
                                'time': start_time * 1000
                            })
                except Exception:
                    continue
    
    if not events:
        return None
    
    return pd.DataFrame(events).sort_values('time').reset_index(drop=True)


def cluster_ripples_by_gap(events_df: pd.DataFrame, max_gap_ms: float = 30.0):
    """Cluster ripples: consecutive events within max_gap_ms belong to same cluster."""
    if len(events_df) == 0:
        return events_df
    
    events_df = events_df.sort_values('time').reset_index(drop=True)
    events_df['cluster_id'] = 0
    
    cluster_id = 0
    prev_time = events_df.iloc[0]['time']
    
    for idx in range(len(events_df)):
        curr_time = events_df.iloc[idx]['time']
        if curr_time - prev_time > max_gap_ms:
            cluster_id += 1
        events_df.at[idx, 'cluster_id'] = cluster_id
        prev_time = curr_time
    
    return events_df


def compute_propagation_metrics(events_df: pd.DataFrame, layout_df: pd.DataFrame):
    """
    For each cluster, compute:
    - Number of channels
    - Max time delay
    - Spatial extent
    - Plane-fit for traveling wave detection
    """
    if 'cluster_id' not in events_df.columns:
        return []
    
    # Create channel-to-position lookup
    ch_to_pos = {}
    for _, row in layout_df.iterrows():
        ch_to_pos[row['bipolar_ch']] = (row['X'], row['Y'])
    
    cluster_metrics = []
    
    for cluster_id in events_df['cluster_id'].unique():
        cluster = events_df[events_df['cluster_id'] == cluster_id]
        
        n_channels = cluster['channel'].nunique()
        channels = cluster['channel'].unique().tolist()
        
        # Get channel positions
        positions = []
        times = []
        for ch in channels:
            if ch in ch_to_pos:
                ch_events = cluster[cluster['channel'] == ch]
                first_time = ch_events['time'].min()
                positions.append(ch_to_pos[ch])
                times.append(first_time)
        
        if len(positions) < 2:
            cluster_metrics.append({
                'cluster_id': cluster_id,
                'n_channels': n_channels,
                'max_delay_ms': 0,
                'spatial_extent': 0,
                'is_traveling': False,
                'plane_fit_r2': 0,
                'plane_fit_p': 1.0,
                'speed_estimate': 0
            })
            continue
        
        positions = np.array(positions)
        times = np.array(times)
        
        # Max time delay
        max_delay = times.max() - times.min()
        
        # Spatial extent (max distance between channels)
        from scipy.spatial.distance import pdist
        spatial_extent = np.max(pdist(positions)) if len(positions) > 1 else 0
        
        # Plane-fit: timing ~ X + Y (traveling wave detection)
        # Normalize times to relative delay from first channel
        relative_times = times - times.min()
        
        # Fit plane: t = a*X + b*Y + c
        if len(positions) >= 3 and max_delay > 0:
            X = np.column_stack([positions[:, 0], positions[:, 1], np.ones(len(positions))])
            try:
                coeffs, residuals, rank, s = np.linalg.lstsq(X, relative_times, rcond=None)
                
                # Predicted times
                predicted = X @ coeffs
                
                # R² 
                ss_res = np.sum((relative_times - predicted) ** 2)
                ss_tot = np.sum((relative_times - relative_times.mean()) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                r2 = max(0, min(1, r2))
                
                # F-test for significance
                n = len(positions)
                p = 2  # X and Y
                if n > p + 1 and ss_tot > 0:
                    f_stat = (r2 / p) / ((1 - r2) / (n - p - 1)) if r2 < 1 else np.inf
                    p_value = 1 - stats.f.cdf(f_stat, p, n - p - 1)
                else:
                    p_value = 1.0
                
                # Speed estimate (ms per unit distance)
                gradient_magnitude = np.sqrt(coeffs[0]**2 + coeffs[1]**2)
                speed = 1 / gradient_magnitude if gradient_magnitude > 0 else 0  # units/ms
                
                is_traveling = r2 > 0.5 and p_value < 0.05 and max_delay > 5
                
            except Exception:
                r2, p_value, speed, is_traveling = 0, 1.0, 0, False
        else:
            r2, p_value, speed = 0, 1.0, 0
            is_traveling = False
        
        cluster_metrics.append({
            'cluster_id': cluster_id,
            'n_channels': n_channels,
            'max_delay_ms': max_delay,
            'spatial_extent': spatial_extent,
            'is_traveling': is_traveling,
            'plane_fit_r2': r2,
            'plane_fit_p': p_value,
            'speed_estimate': speed
        })
    
    return cluster_metrics


def analyze_session_propagation(session: int, trial: int = 1, z_low: float = 2.5):
    """Run propagation analysis for a single session."""
    base = Path("/vol/brains/bd3/pesaranlab/Archie_RecStim_vSUBNETS220_2nd/matlab/mfiles/Chen")
    trial_dir = base / f"session{session:03d}" / f"trial{trial:03d}_bipolar"
    
    if not trial_dir.exists():
        return None
    
    # Load layout
    layout_file = trial_dir / "bipolar_layout.csv"
    if not layout_file.exists():
        layout_file = base / "bipolar_layout.csv"  # Fallback to common layout
    
    if not layout_file.exists():
        print(f"  ⚠ No layout file found for session {session}")
        return None
    
    layout_df = pd.read_csv(layout_file)
    
    # Load ripple events
    events_df = load_ripple_events(trial_dir, z_low)
    if events_df is None or len(events_df) == 0:
        print(f"  ⚠ No ripple events found for session {session}")
        return None
    
    print(f"  Loaded {len(events_df)} ripple events across {events_df['channel'].nunique()} channels")
    
    # Cluster by gap
    events_df = cluster_ripples_by_gap(events_df, max_gap_ms=30.0)
    n_clusters = events_df['cluster_id'].nunique()
    print(f"  Found {n_clusters} clusters (gap=30ms)")
    
    # Compute propagation metrics
    metrics = compute_propagation_metrics(events_df, layout_df)
    metrics_df = pd.DataFrame(metrics)
    
    # Summarize
    multi_channel = metrics_df[metrics_df['n_channels'] > 1]
    traveling = metrics_df[metrics_df['is_traveling']]
    
    summary = {
        'session': session,
        'trial': trial,
        'n_events': len(events_df),
        'n_clusters': n_clusters,
        'n_single_channel': len(metrics_df[metrics_df['n_channels'] == 1]),
        'n_multi_channel': len(multi_channel),
        'n_traveling': len(traveling),
        'pct_multi_channel': 100 * len(multi_channel) / n_clusters if n_clusters > 0 else 0,
        'pct_traveling': 100 * len(traveling) / n_clusters if n_clusters > 0 else 0,
        'mean_delay_multi': multi_channel['max_delay_ms'].mean() if len(multi_channel) > 0 else 0,
        'mean_r2_multi': multi_channel['plane_fit_r2'].mean() if len(multi_channel) > 0 else 0,
    }
    
    # Save results
    output_dir = trial_dir / "propagation_analysis"
    output_dir.mkdir(exist_ok=True)
    
    metrics_df.to_csv(output_dir / f"cluster_metrics_zlow{z_low}.csv", index=False)
    
    return summary


def main():
    """Run propagation analysis on all processed sessions."""
    
    print("=" * 80)
    print("PROPAGATION ANALYSIS: Testing for Traveling Waves")
    print("=" * 80)
    print("\nHypothesis: Sessions without local synchrony might have propagation!")
    print()
    
    # Sessions to analyze
    sessions = [129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 143, 144]
    trials = {sess: 1 if sess != 144 else 6 for sess in sessions}
    
    # Track synchrony status
    sync_sessions = [131, 132, 133, 134]  # Known from previous analysis
    
    results = []
    
    for sess in sessions:
        trial = trials[sess]
        has_sync = sess in sync_sessions
        
        status = "SYNCHRONY" if has_sync else "NO SYNC"
        print(f"\n[Session {sess}] ({status})")
        print("-" * 40)
        
        summary = analyze_session_propagation(sess, trial)
        if summary:
            summary['has_synchrony'] = has_sync
            results.append(summary)
    
    # Create comparison table
    print("\n" + "=" * 100)
    print("PROPAGATION SUMMARY: Synchrony vs Non-Synchrony Sessions")
    print("=" * 100)
    
    if not results:
        print("\n⚠ No sessions had ripple data to analyze!")
        return
    
    df = pd.DataFrame(results)
    
    print(f"\n{'Session':<10} {'Sync?':<8} {'Clusters':<10} {'Multi-Ch':<10} {'Traveling':<10} {'%Travel':<10} {'Mean R²':<10}")
    print("-" * 78)
    
    for _, row in df.iterrows():
        sync_str = "YES" if row['has_synchrony'] else "NO"
        pct_travel = f"{row['pct_traveling']:.1f}%"
        mean_r2 = f"{row['mean_r2_multi']:.3f}"
        print(f"{row['session']:<10} {sync_str:<8} {row['n_clusters']:<10} {row['n_multi_channel']:<10} {row['n_traveling']:<10} {pct_travel:<10} {mean_r2:<10}")
    
    # Compare groups
    print("\n" + "=" * 80)
    print("GROUP COMPARISON")
    print("=" * 80)
    
    sync_df = df[df['has_synchrony']]
    nosync_df = df[~df['has_synchrony']]
    
    print(f"\n{'Metric':<30} {'With Sync (N={len(sync_df)})':<25} {'Without Sync (N={len(nosync_df)})':<25}")
    print("-" * 80)
    
    for metric in ['pct_multi_channel', 'pct_traveling', 'mean_delay_multi', 'mean_r2_multi']:
        if metric in df.columns:
            sync_mean = sync_df[metric].mean()
            nosync_mean = nosync_df[metric].mean()
            
            # Statistical test
            if len(sync_df) > 1 and len(nosync_df) > 1:
                stat, p_val = stats.mannwhitneyu(sync_df[metric], nosync_df[metric], alternative='two-sided')
                p_str = f"p={p_val:.4f}" if p_val >= 0.0001 else "p<0.0001"
            else:
                p_str = "N/A"
            
            print(f"{metric:<30} {sync_mean:<25.2f} {nosync_mean:<25.2f} {p_str}")
    
    # Key interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    
    # Test hypothesis: Do non-sync sessions have MORE traveling waves?
    if len(sync_df) > 0 and len(nosync_df) > 0:
        sync_pct_travel = sync_df['pct_traveling'].mean()
        nosync_pct_travel = nosync_df['pct_traveling'].mean()
        
        stat, p_val = stats.mannwhitneyu(
            nosync_df['pct_traveling'], 
            sync_df['pct_traveling'], 
            alternative='greater'  # One-sided: nosync > sync
        )
        
        print(f"\nHypothesis: Sessions without synchrony have MORE propagation")
        print(f"  - With synchrony:    {sync_pct_travel:.2f}% traveling clusters")
        print(f"  - Without synchrony: {nosync_pct_travel:.2f}% traveling clusters")
        print(f"  - Mann-Whitney U test (one-sided, nosync > sync): p = {p_val:.4f}")
        
        if p_val < 0.05:
            print(f"\n  ✓ SUPPORTED: Non-synchrony sessions have significantly more propagation!")
        elif nosync_pct_travel > sync_pct_travel:
            print(f"\n  ~ TREND: Non-synchrony sessions show more propagation (not significant)")
        else:
            print(f"\n  ✗ NOT SUPPORTED: No evidence of more propagation in non-synchrony sessions")
    
    # Save summary
    base = Path("/vol/brains/bd3/pesaranlab/Archie_RecStim_vSUBNETS220_2nd/matlab/mfiles/Chen")
    df.to_csv(base / "propagation_summary_all_sessions.csv", index=False)
    print(f"\nSaved: {base}/propagation_summary_all_sessions.csv")


if __name__ == "__main__":
    main()
