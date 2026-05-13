#!/usr/bin/env python3
"""
Propagation Analysis: Detect Traveling Waves in Ripple Events

Analyzes whether ripples propagate across the cortical surface as traveling waves.

Methodology (v2 - Recommended):
  1. Anchor-based event definition: For each ripple, define ±window_ms
  2. Match closest ripple from each other channel within window
  3. Fit plane to timing delays (t = aX + bY + c)
  4. Compare to shuffled null for significance

Alternative (v1 - Legacy):
  1. Gap-based clustering of ripples
  2. Plane-fit for multi-channel clusters
  3. Fixed R² threshold for significance

Usage Examples:
    # Single session analysis
    python run_propagation_analysis.py --session 134 --trial 1
    
    # Batch mode (all sessions)
    python run_propagation_analysis.py --batch
    
    # Legacy v1 method
    python run_propagation_analysis.py --session 134 --trial 1 --method v1
    
    # Run diagnostics
    python run_propagation_analysis.py --session 134 --diagnostics
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
from scipy.spatial.distance import pdist
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import argparse
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'packages' / 'ripple_core'))


# ============================================================================
# Configuration
# ============================================================================
BASE_DIR = Path("/vol/brains/bd3/pesaranlab/Archie_RecStim_vSUBNETS220_2nd/matlab/mfiles/Chen")
DEFAULT_Z_LOW = 3.0
DEFAULT_WINDOW_MS = 10.0
DEFAULT_MIN_CHANNELS = 4
DEFAULT_N_SHUFFLES = 500


# ============================================================================
# Data Loading
# ============================================================================
def load_ripple_events(trial_dir: Path, z_low: float = DEFAULT_Z_LOW,
                       include_peaks: bool = True) -> pd.DataFrame:
    """
    Load ripple events from all bipolar channels.
    
    Args:
        trial_dir: Path to trial_bipolar directory
        z_low: Z-score threshold used for detection
        include_peaks: Include peak times (for v2 method)
    
    Returns:
        DataFrame with columns: channel, start_ms, peak_ms (if include_peaks)
    """
    events = []
    
    for ch_dir in sorted(trial_dir.glob("b[0-9][0-9][0-9]")):
        if not ch_dir.is_dir():
            continue
        channel = ch_dir.name
        
        # Try multiple naming patterns
        patterns = [
            f"ripples_{channel}_zlow{z_low}.mat",
            f"ripples_{channel}_zlow3.0.mat",
            f"ripples_{channel}_zlow2.5.mat",
            f"ripples_{channel}.mat",
        ]
        
        mat_file = None
        for pattern in patterns:
            f = ch_dir / pattern
            if f.exists():
                mat_file = f
                break
        
        if mat_file is None:
            continue
        
                try:
                    data = sio.loadmat(str(mat_file), squeeze_me=True)
            fs = float(data.get('fs', 1000.0))
                    
            # Get starts
                    starts = None
            for key in ['merged_starts', 'starts', 'start_times']:
                        if key in data and data[key] is not None:
                            starts = np.atleast_1d(data[key])
                            break
                    
            if starts is None or len(starts) == 0:
                    continue
            
            # Get peaks if requested
            peaks = None
            if include_peaks:
                for key in ['peak_idx', 'merged_peaks', 'peaks']:
                    if key in data and data[key] is not None:
                        peaks = np.atleast_1d(data[key])
                break
    
                # Get ends for estimating peaks
                ends = None
                for key in ['merged_ends', 'ends']:
                        if key in data and data[key] is not None:
                        ends = np.atleast_1d(data[key])
                            break
                    
                # Estimate peaks if not available
                if peaks is None and ends is not None and len(ends) == len(starts):
                    peaks = (starts + ends) // 2
                elif peaks is None:
                    peaks = starts
                
                # Ensure same length
                n = min(len(starts), len(peaks))
                starts = starts[:n]
                peaks = peaks[:n]
            
            # Convert to milliseconds and add events
            for i in range(len(starts)):
                event = {
                                'channel': channel,
                    'start_ms': starts[i] / fs * 1000,
                }
                if include_peaks:
                    event['peak_ms'] = peaks[i] / fs * 1000
                events.append(event)
                
        except Exception as e:
                    continue
    
    if not events:
        return None
    
    time_col = 'peak_ms' if include_peaks else 'start_ms'
    return pd.DataFrame(events).sort_values(time_col).reset_index(drop=True)


def load_layout(trial_dir: Path) -> pd.DataFrame:
    """Load electrode layout."""
    layout_file = trial_dir / "bipolar_layout.csv"
    if not layout_file.exists():
        layout_file = BASE_DIR / "bipolar_layout.csv"
    
    if not layout_file.exists():
        return None
    
    return pd.read_csv(layout_file)


# ============================================================================
# V2 Method: Anchor-Based Event Definition
# ============================================================================
def define_events_anchor_based(events_df: pd.DataFrame, 
                                window_ms: float = DEFAULT_WINDOW_MS,
                                min_channels: int = DEFAULT_MIN_CHANNELS) -> list:
    """
    Define multi-channel events using anchor-based approach.
    
    For each ripple (anchor), find the closest ripple from each other channel
    within ±window_ms. This creates proper "events" with one ripple per channel.
    
    Returns list of events, each is a dict with:
        - anchor_channel, anchor_time
        - participants: list of (channel, peak_time) tuples
        - n_channels
    """
    channels = events_df['channel'].unique()
    events = []
    used_indices = set()
    
    df = events_df.sort_values('peak_ms').reset_index(drop=True)
    
    for idx, anchor_row in df.iterrows():
        if idx in used_indices:
            continue
        
        anchor_ch = anchor_row['channel']
        anchor_time = anchor_row['peak_ms']
        
        participants = [(anchor_ch, anchor_time)]
        participant_indices = [idx]
        
        for ch in channels:
            if ch == anchor_ch:
                continue
            
            ch_ripples = df[df['channel'] == ch]
            in_window = ch_ripples[
                (ch_ripples['peak_ms'] >= anchor_time - window_ms) &
                (ch_ripples['peak_ms'] <= anchor_time + window_ms)
            ]
            
            if len(in_window) > 0:
                closest_idx = (in_window['peak_ms'] - anchor_time).abs().idxmin()
                closest_time = in_window.loc[closest_idx, 'peak_ms']
                participants.append((ch, closest_time))
                participant_indices.append(closest_idx)
        
        if len(participants) >= min_channels:
            events.append({
                'anchor_channel': anchor_ch,
                'anchor_time': anchor_time,
                'participants': participants,
                'n_channels': len(participants)
            })
            used_indices.update(participant_indices)
    
    return events


def compute_propagation_metrics_v2(event: dict, layout_df: pd.DataFrame) -> dict:
    """Compute propagation metrics for a single event using plane-fit."""
    participants = event['participants']
    
    ch_to_pos = {}
    for _, row in layout_df.iterrows():
        ch_to_pos[row['bipolar_ch']] = (row['X'], row['Y'])
    
    positions = []
    times = []
    for ch, t in participants:
        if ch in ch_to_pos:
            positions.append(ch_to_pos[ch])
            times.append(t)
    
    if len(positions) < 3:
        return None
    
    positions = np.array(positions)
    times = np.array(times)
    
    relative_times = times - times.min()
    max_delay = relative_times.max()
    spatial_extent = np.max(pdist(positions)) if len(positions) > 1 else 0
    
    # Plane fit: t = a*X + b*Y + c
    X = np.column_stack([positions[:, 0], positions[:, 1], np.ones(len(positions))])
    
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, relative_times, rcond=None)
        predicted = X @ coeffs
        
        ss_res = np.sum((relative_times - predicted) ** 2)
        ss_tot = np.sum((relative_times - relative_times.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        r2 = max(0, min(1, r2))
        
        gradient = coeffs[:2]
        gradient_mag = np.sqrt(gradient[0]**2 + gradient[1]**2)
        speed = 1 / gradient_mag if gradient_mag > 0 else 0
        direction = np.degrees(np.arctan2(gradient[1], gradient[0]))
        
    except Exception:
        return None
    
    return {
        'n_channels': len(positions),
        'max_delay_ms': max_delay,
        'spatial_extent': spatial_extent,
        'r2': r2,
        'speed': speed,
        'direction': direction,
        'times': times,
        'positions': positions
    }


def compute_null_r2_distribution(positions: np.ndarray, times: np.ndarray,
                                  n_shuffles: int = DEFAULT_N_SHUFFLES) -> np.ndarray:
    """Compute null distribution of R² by shuffling times."""
    null_r2s = []
    
    for _ in range(n_shuffles):
        shuffled_times = times.copy()
        np.random.shuffle(shuffled_times)
        
        relative_times = shuffled_times - shuffled_times.min()
        
        X = np.column_stack([positions[:, 0], positions[:, 1], np.ones(len(positions))])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, relative_times, rcond=None)
            predicted = X @ coeffs
            
            ss_res = np.sum((relative_times - predicted) ** 2)
            ss_tot = np.sum((relative_times - relative_times.mean()) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            r2 = max(0, min(1, r2))
            null_r2s.append(r2)
        except:
            null_r2s.append(0)
    
    return np.array(null_r2s)


# ============================================================================
# V1 Method: Gap-Based Clustering (Legacy)
# ============================================================================
def cluster_ripples_by_gap(events_df: pd.DataFrame, max_gap_ms: float = 30.0) -> pd.DataFrame:
    """Cluster ripples: consecutive events within max_gap_ms belong to same cluster."""
    if len(events_df) == 0:
        return events_df
    
    events_df = events_df.sort_values('start_ms').reset_index(drop=True)
    events_df['cluster_id'] = 0
    
    cluster_id = 0
    prev_time = events_df.iloc[0]['start_ms']
    
    for idx in range(len(events_df)):
        curr_time = events_df.iloc[idx]['start_ms']
        if curr_time - prev_time > max_gap_ms:
            cluster_id += 1
        events_df.at[idx, 'cluster_id'] = cluster_id
        prev_time = curr_time
    
    return events_df


def compute_propagation_metrics_v1(events_df: pd.DataFrame, layout_df: pd.DataFrame) -> list:
    """Compute metrics for each cluster (v1 method)."""
    if 'cluster_id' not in events_df.columns:
        return []
    
    ch_to_pos = {}
    for _, row in layout_df.iterrows():
        ch_to_pos[row['bipolar_ch']] = (row['X'], row['Y'])
    
    cluster_metrics = []
    
    for cluster_id in events_df['cluster_id'].unique():
        cluster = events_df[events_df['cluster_id'] == cluster_id]
        n_channels = cluster['channel'].nunique()
        channels = cluster['channel'].unique().tolist()
        
        positions = []
        times = []
        for ch in channels:
            if ch in ch_to_pos:
                ch_events = cluster[cluster['channel'] == ch]
                first_time = ch_events['start_ms'].min()
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
        
        max_delay = times.max() - times.min()
        spatial_extent = np.max(pdist(positions)) if len(positions) > 1 else 0
        
        relative_times = times - times.min()
        
        if len(positions) >= 3 and max_delay > 0:
            X = np.column_stack([positions[:, 0], positions[:, 1], np.ones(len(positions))])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, relative_times, rcond=None)
                predicted = X @ coeffs
                
                ss_res = np.sum((relative_times - predicted) ** 2)
                ss_tot = np.sum((relative_times - relative_times.mean()) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                r2 = max(0, min(1, r2))
                
                n = len(positions)
                p = 2
                if n > p + 1 and ss_tot > 0:
                    f_stat = (r2 / p) / ((1 - r2) / (n - p - 1)) if r2 < 1 else np.inf
                    p_value = 1 - stats.f.cdf(f_stat, p, n - p - 1)
                else:
                    p_value = 1.0
                
                gradient_mag = np.sqrt(coeffs[0]**2 + coeffs[1]**2)
                speed = 1 / gradient_mag if gradient_mag > 0 else 0
                
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


# ============================================================================
# Main Analysis Functions
# ============================================================================
def analyze_session(session: int, trial: int = 1, z_low: float = DEFAULT_Z_LOW,
                    method: str = 'v2', window_ms: float = DEFAULT_WINDOW_MS,
                    min_channels: int = DEFAULT_MIN_CHANNELS) -> dict:
    """Run propagation analysis for a single session."""
    
    trial_dir = BASE_DIR / f"session{session:03d}" / f"trial{trial:03d}_bipolar"
    
    if not trial_dir.exists():
        print(f"  ⚠ Trial directory not found: {trial_dir}")
        return None
    
    layout_df = load_layout(trial_dir)
    if layout_df is None:
        print(f"  ⚠ No layout file found")
        return None
    
    # Load events
    print(f"  Loading ripples...")
    events_df = load_ripple_events(trial_dir, z_low, include_peaks=(method == 'v2'))
    if events_df is None or len(events_df) == 0:
        print(f"  ⚠ No ripple events found")
        return None
    
    n_ripples = len(events_df)
    n_channels = events_df['channel'].nunique()
    print(f"  Found {n_ripples} ripples across {n_channels} channels")
    
    if method == 'v2':
        return _analyze_v2(events_df, layout_df, trial_dir, session, trial,
                          window_ms, min_channels, z_low)
    else:
        return _analyze_v1(events_df, layout_df, trial_dir, session, trial, z_low)


def _analyze_v2(events_df, layout_df, trial_dir, session, trial,
                window_ms, min_channels, z_low):
    """V2 analysis with anchor-based events."""
    print(f"  Defining events (window=±{window_ms}ms, min_ch={min_channels})...")
    events = define_events_anchor_based(events_df, window_ms, min_channels)
    print(f"  Found {len(events)} multi-channel events")
    
    if len(events) == 0:
        return {
            'session': session, 'trial': trial, 'method': 'v2',
            'n_ripples': len(events_df), 'n_channels': events_df['channel'].nunique(),
            'n_events': 0, 'n_traveling': 0, 'pct_traveling': 0,
            'mean_r2': 0, 'mean_delay': 0
        }
    
    results = []
    n_traveling = 0
    
    for event in events:
        metrics = compute_propagation_metrics_v2(event, layout_df)
        if metrics is None:
            continue
        
        null_r2s = compute_null_r2_distribution(metrics['positions'], metrics['times'])
        p_value = np.mean(null_r2s >= metrics['r2'])
        
        metrics['p_value'] = p_value
        metrics['is_traveling'] = (p_value < 0.05) and (metrics['max_delay_ms'] > 2)
        
        if metrics['is_traveling']:
            n_traveling += 1
        
        results.append(metrics)
    
    if results:
        mean_r2 = np.mean([r['r2'] for r in results])
        mean_delay = np.mean([r['max_delay_ms'] for r in results])
        pct_traveling = 100 * n_traveling / len(results)
    else:
        mean_r2, mean_delay, pct_traveling = 0, 0, 0
    
    # Save results
    output_dir = trial_dir / "propagation_analysis"
    output_dir.mkdir(exist_ok=True)
    
    results_df = pd.DataFrame([{
        'n_channels': r['n_channels'],
        'max_delay_ms': r['max_delay_ms'],
        'spatial_extent': r['spatial_extent'],
        'r2': r['r2'],
        'p_value': r['p_value'],
        'speed': r['speed'],
        'direction': r['direction'],
        'is_traveling': r['is_traveling']
    } for r in results])
    
    results_df.to_csv(output_dir / f"event_metrics_zlow{z_low}.csv", index=False)
    
    print(f"  Traveling waves: {n_traveling}/{len(results)} ({pct_traveling:.1f}%)")
    
    return {
        'session': session, 'trial': trial, 'method': 'v2',
        'n_ripples': len(events_df), 'n_channels': events_df['channel'].nunique(),
        'n_events': len(events), 'n_traveling': n_traveling,
        'pct_traveling': pct_traveling, 'mean_r2': mean_r2, 'mean_delay': mean_delay
    }


def _analyze_v1(events_df, layout_df, trial_dir, session, trial, z_low):
    """V1 analysis with gap-based clustering."""
    events_df = cluster_ripples_by_gap(events_df, max_gap_ms=30.0)
    n_clusters = events_df['cluster_id'].nunique()
    print(f"  Found {n_clusters} clusters (gap=30ms)")
    
    metrics = compute_propagation_metrics_v1(events_df, layout_df)
    metrics_df = pd.DataFrame(metrics)
    
    multi_channel = metrics_df[metrics_df['n_channels'] > 1]
    traveling = metrics_df[metrics_df['is_traveling']]
    
    # Save results
    output_dir = trial_dir / "propagation_analysis"
    output_dir.mkdir(exist_ok=True)
    metrics_df.to_csv(output_dir / f"cluster_metrics_zlow{z_low}.csv", index=False)
    
    return {
        'session': session, 'trial': trial, 'method': 'v1',
        'n_events': len(events_df), 'n_clusters': n_clusters,
        'n_single_channel': len(metrics_df[metrics_df['n_channels'] == 1]),
        'n_multi_channel': len(multi_channel),
        'n_traveling': len(traveling),
        'pct_multi_channel': 100 * len(multi_channel) / n_clusters if n_clusters > 0 else 0,
        'pct_traveling': 100 * len(traveling) / n_clusters if n_clusters > 0 else 0,
        'mean_delay_multi': multi_channel['max_delay_ms'].mean() if len(multi_channel) > 0 else 0,
        'mean_r2_multi': multi_channel['plane_fit_r2'].mean() if len(multi_channel) > 0 else 0,
    }
    

def run_diagnostics(session: int, trial: int, z_low: float):
    """Run diagnostic checks on the data."""
    trial_dir = BASE_DIR / f"session{session:03d}" / f"trial{trial:03d}_bipolar"
    
    print("="*60)
    print("DIAGNOSTICS")
    print("="*60)
    
    events_df = load_ripple_events(trial_dir, z_low, include_peaks=True)
    if events_df is None:
        print("No events found!")
        return
    
    print(f"\n1. TIME UNIT CHECK:")
    print(f"   Min peak time: {events_df['peak_ms'].min():.2f} ms")
    print(f"   Max peak time: {events_df['peak_ms'].max():.2f} ms")
    print(f"   Expected for 5-min trial: 0 - 300,000 ms")
    
    max_time = events_df['peak_ms'].max()
    if max_time > 1e6:
        print("   ⚠ Times too large - check units")
    elif max_time < 1000:
        print("   ⚠ Times too small - check units")
    else:
        print("   ✓ Times look reasonable")
    
    print(f"\n2. INTER-RIPPLE INTERVALS:")
    df_sorted = events_df.sort_values('peak_ms')
    iris = df_sorted['peak_ms'].diff().dropna()
    print(f"   Median IRI: {iris.median():.1f} ms")
    print(f"   Mean IRI: {iris.mean():.1f} ms")
    print(f"   Fraction < 30ms: {100 * (iris < 30).mean():.1f}%")
    
    print(f"\n3. RIPPLES PER CHANNEL:")
    counts = events_df['channel'].value_counts()
    print(f"   Min: {counts.min()}, Max: {counts.max()}, Median: {counts.median():.0f}")


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Propagation Analysis for Ripple Events',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single session (v2 method - recommended)
    python run_propagation_analysis.py --session 134 --trial 1
    
    # Legacy v1 method
    python run_propagation_analysis.py --session 134 --method v1
    
    # Batch mode
    python run_propagation_analysis.py --batch --trial 1
    
    # Diagnostics
    python run_propagation_analysis.py --session 134 --diagnostics
        """
    )
    
    parser.add_argument('--session', type=int, help='Session number')
    parser.add_argument('--trial', type=int, default=1, help='Trial number')
    parser.add_argument('--z_low', type=float, default=DEFAULT_Z_LOW)
    parser.add_argument('--method', choices=['v1', 'v2'], default='v2',
                       help='Analysis method (v2 recommended)')
    parser.add_argument('--window_ms', type=float, default=DEFAULT_WINDOW_MS,
                       help='Event window size in ms (v2 only)')
    parser.add_argument('--min_channels', type=int, default=DEFAULT_MIN_CHANNELS,
                       help='Min channels for multi-channel event')
    parser.add_argument('--diagnostics', action='store_true',
                       help='Run diagnostics only')
    parser.add_argument('--batch', action='store_true',
                       help='Run on all sessions with ripple data')
    
    args = parser.parse_args()
    
    if args.diagnostics and args.session:
        run_diagnostics(args.session, args.trial, args.z_low)
        return
    
    if args.session and not args.batch:
        print("="*80)
        print(f"PROPAGATION ANALYSIS ({args.method.upper()})")
        print(f"Session {args.session}, Trial {args.trial}")
        print("="*80)
        
        result = analyze_session(
            args.session, args.trial, args.z_low,
            method=args.method, window_ms=args.window_ms,
            min_channels=args.min_channels
        )
        
        if result:
            print(f"\nSUMMARY:")
            print(f"  Events: {result.get('n_events', result.get('n_clusters', 0))}")
            print(f"  Traveling: {result['n_traveling']} ({result['pct_traveling']:.1f}%)")
            print(f"  Mean R²: {result.get('mean_r2', result.get('mean_r2_multi', 0)):.3f}")
        return
    
    if args.batch:
        print("="*80)
        print(f"PROPAGATION ANALYSIS ({args.method.upper()}) - BATCH MODE")
        print("="*80)
        
        sessions = []
        for d in sorted(BASE_DIR.glob("session[0-9][0-9][0-9]")):
            sess_num = int(d.name.replace("session", ""))
            trial_dir = d / f"trial{args.trial:03d}_bipolar"
            if trial_dir.exists() and any(trial_dir.glob("b[0-9][0-9][0-9]/ripples_*.mat")):
                sessions.append(sess_num)
        
        print(f"Found {len(sessions)} sessions with ripple data")
        
        results = []
        for sess in sessions:
            print(f"\n[Session {sess}]")
            result = analyze_session(
                sess, args.trial, args.z_low,
                method=args.method, window_ms=args.window_ms,
                min_channels=args.min_channels
            )
            if result:
                results.append(result)
        
        if results:
            df = pd.DataFrame(results)
            
            print("\n" + "="*80)
            print("BATCH SUMMARY")
            print("="*80)
            print(f"\n{'Session':<10} {'Events':<10} {'Travel':<10} {'%':<8} {'R²':<8}")
            print("-"*50)
            
            for _, row in df.iterrows():
                n_events = row.get('n_events', row.get('n_clusters', 0))
                print(f"{row['session']:<10} {n_events:<10} {row['n_traveling']:<10} "
                      f"{row['pct_traveling']:.1f}%{'':>4} "
                      f"{row.get('mean_r2', row.get('mean_r2_multi', 0)):.3f}")
            
            df.to_csv(BASE_DIR / f"propagation_summary_{args.method}.csv", index=False)
            print(f"\nSaved: propagation_summary_{args.method}.csv")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
