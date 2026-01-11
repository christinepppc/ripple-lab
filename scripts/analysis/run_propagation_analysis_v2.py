#!/usr/bin/env python3
"""
Propagation Analysis v2: Detect traveling waves in ripple events.

FIXES from v1:
1. Anchor-based event definition (not gap-chaining)
2. Use peak times (envelope max), not start times
3. Permutation null for significance (not arbitrary R² threshold)
4. Proper time unit validation

For each session:
1. For each ripple, define event window (±10ms)
2. Match closest ripple from each other channel within window
3. Fit plane to timing delays
4. Compare to shuffled null
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
from scipy.signal import hilbert, butter, filtfilt
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add packages to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'packages' / 'ripple_core'))


def load_ripple_events_with_peaks(trial_dir: Path, z_low: float = 2.5):
    """
    Load ripple events with peak times (not just starts).
    
    Returns DataFrame with columns: channel, start_ms, peak_ms
    """
    events = []
    
    for ch_dir in sorted(trial_dir.glob("b[0-9][0-9][0-9]")):
        if not ch_dir.is_dir():
            continue
        channel = ch_dir.name
        
        # Find ripple file
        patterns = [
            f"ripples_{channel}_zlow{z_low}.mat",
            f"ripples_{channel}_zlow2.5.mat",
            f"ripples_{channel}_zlow3.0.mat",
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
            
            # Get starts (in samples)
            starts = None
            for key in ['merged_starts', 'starts']:
                if key in data and data[key] is not None:
                    starts = np.atleast_1d(data[key])
                    break
            
            # Get peaks (in samples) - prefer peak_idx if available
            peaks = None
            for key in ['peak_idx', 'merged_peaks', 'peaks']:
                if key in data and data[key] is not None:
                    peaks = np.atleast_1d(data[key])
                    break
            
            # Get ends for computing peak from envelope if needed
            ends = None
            for key in ['merged_ends', 'ends']:
                if key in data and data[key] is not None:
                    ends = np.atleast_1d(data[key])
                    break
            
            if starts is None or len(starts) == 0:
                continue
            
            # If no peaks, estimate as midpoint between start and end
            if peaks is None and ends is not None and len(ends) == len(starts):
                peaks = (starts + ends) // 2
            elif peaks is None:
                peaks = starts  # Fallback to starts
            
            # Ensure same length
            n = min(len(starts), len(peaks))
            starts = starts[:n]
            peaks = peaks[:n]
            
            # Convert to milliseconds
            for i in range(n):
                start_ms = starts[i] / fs * 1000
                peak_ms = peaks[i] / fs * 1000
                
                # Sanity check: peak should be within reasonable range
                if 0 < start_ms < 1e6 and 0 < peak_ms < 1e6:  # Less than ~16 min
                    events.append({
                        'channel': channel,
                        'start_ms': start_ms,
                        'peak_ms': peak_ms
                    })
        except Exception as e:
            continue
    
    if not events:
        return None
    
    df = pd.DataFrame(events).sort_values('peak_ms').reset_index(drop=True)
    
    # Sanity check: print time range
    print(f"    Time range: {df['peak_ms'].min():.1f} - {df['peak_ms'].max():.1f} ms")
    print(f"    Duration: {(df['peak_ms'].max() - df['peak_ms'].min()) / 1000:.1f} sec")
    
    return df


def define_events_anchor_based(events_df: pd.DataFrame, window_ms: float = 10.0, 
                                min_channels: int = 3):
    """
    Define multi-channel events using anchor-based approach.
    
    For each ripple (anchor), find the closest ripple from each other channel
    within ±window_ms. This creates a proper "event" with one ripple per channel.
    
    Returns list of events, each is a dict with:
        - anchor_channel, anchor_time
        - participants: list of (channel, peak_time) tuples
    """
    channels = events_df['channel'].unique()
    events = []
    used_indices = set()  # Track which ripples have been used as anchors
    
    # Sort by peak time
    df = events_df.sort_values('peak_ms').reset_index(drop=True)
    
    for idx, anchor_row in df.iterrows():
        if idx in used_indices:
            continue
            
        anchor_ch = anchor_row['channel']
        anchor_time = anchor_row['peak_ms']
        
        # Find closest ripple from each other channel within window
        participants = [(anchor_ch, anchor_time)]
        participant_indices = [idx]
        
        for ch in channels:
            if ch == anchor_ch:
                continue
            
            # Get all ripples from this channel
            ch_ripples = df[df['channel'] == ch]
            
            # Find those within window
            in_window = ch_ripples[
                (ch_ripples['peak_ms'] >= anchor_time - window_ms) &
                (ch_ripples['peak_ms'] <= anchor_time + window_ms)
            ]
            
            if len(in_window) > 0:
                # Take the closest one
                closest_idx = (in_window['peak_ms'] - anchor_time).abs().idxmin()
                closest_time = in_window.loc[closest_idx, 'peak_ms']
                participants.append((ch, closest_time))
                participant_indices.append(closest_idx)
        
        # Only keep events with enough channels
        if len(participants) >= min_channels:
            events.append({
                'anchor_channel': anchor_ch,
                'anchor_time': anchor_time,
                'participants': participants,
                'n_channels': len(participants)
            })
            # Mark all participant ripples as used (to avoid double-counting)
            used_indices.update(participant_indices)
    
    return events


def compute_propagation_for_event(event: dict, layout_df: pd.DataFrame):
    """
    For a single event, compute propagation metrics using plane-fit.
    
    Returns dict with R², delay stats, etc.
    """
    participants = event['participants']
    
    # Create channel-to-position lookup
    ch_to_pos = {}
    for _, row in layout_df.iterrows():
        ch_to_pos[row['bipolar_ch']] = (row['X'], row['Y'])
    
    # Get positions and times for participants
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
    
    # Relative times from earliest
    relative_times = times - times.min()
    max_delay = relative_times.max()
    
    # Compute spatial extent
    from scipy.spatial.distance import pdist
    spatial_extent = np.max(pdist(positions))
    
    # Plane fit: t = a*X + b*Y + c
    X = np.column_stack([positions[:, 0], positions[:, 1], np.ones(len(positions))])
    
    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(X, relative_times, rcond=None)
        predicted = X @ coeffs
        
        ss_res = np.sum((relative_times - predicted) ** 2)
        ss_tot = np.sum((relative_times - relative_times.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        r2 = max(0, min(1, r2))
        
        # Gradient (direction and speed)
        gradient = coeffs[:2]  # [a, b] = dt/dx, dt/dy
        gradient_mag = np.sqrt(gradient[0]**2 + gradient[1]**2)
        
        # Speed in distance units per ms
        speed = 1 / gradient_mag if gradient_mag > 0 else 0
        
        # Direction (angle in degrees)
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
                                  n_shuffles: int = 500):
    """
    Compute null distribution of R² by shuffling times across channels.
    """
    null_r2s = []
    
    for _ in range(n_shuffles):
        # Shuffle times
        shuffled_times = times.copy()
        np.random.shuffle(shuffled_times)
        
        relative_times = shuffled_times - shuffled_times.min()
        
        # Plane fit
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


def analyze_session_propagation_v2(session: int, trial: int = 1, z_low: float = 2.5,
                                    window_ms: float = 10.0, min_channels: int = 4):
    """Run improved propagation analysis for a single session."""
    
    base = Path("/vol/brains/bd3/pesaranlab/Archie_RecStim_vSUBNETS220_2nd/matlab/mfiles/Chen")
    trial_dir = base / f"session{session:03d}" / f"trial{trial:03d}_bipolar"
    
    if not trial_dir.exists():
        print(f"  ⚠ Trial directory not found: {trial_dir}")
        return None
    
    # Load layout
    layout_file = trial_dir / "bipolar_layout.csv"
    if not layout_file.exists():
        layout_file = base / "bipolar_layout.csv"
    if not layout_file.exists():
        print(f"  ⚠ No layout file found")
        return None
    
    layout_df = pd.read_csv(layout_file)
    
    # Load ripple events with peaks
    print(f"  Loading ripples...")
    events_df = load_ripple_events_with_peaks(trial_dir, z_low)
    if events_df is None or len(events_df) == 0:
        print(f"  ⚠ No ripple events found")
        return None
    
    n_ripples = len(events_df)
    n_channels = events_df['channel'].nunique()
    print(f"  Found {n_ripples} ripples across {n_channels} channels")
    
    # Define events using anchor-based approach
    print(f"  Defining events (window=±{window_ms}ms, min_ch={min_channels})...")
    events = define_events_anchor_based(events_df, window_ms=window_ms, 
                                         min_channels=min_channels)
    print(f"  Found {len(events)} multi-channel events")
    
    if len(events) == 0:
        return {
            'session': session,
            'trial': trial,
            'n_ripples': n_ripples,
            'n_channels': n_channels,
            'n_events': 0,
            'n_traveling': 0,
            'pct_traveling': 0,
            'mean_r2': 0,
            'mean_delay': 0
        }
    
    # Analyze each event
    results = []
    n_traveling = 0
    
    for event in events:
        metrics = compute_propagation_for_event(event, layout_df)
        if metrics is None:
            continue
        
        # Compute null distribution and p-value
        null_r2s = compute_null_r2_distribution(metrics['positions'], metrics['times'])
        p_value = np.mean(null_r2s >= metrics['r2'])
        
        metrics['p_value'] = p_value
        metrics['is_traveling'] = (p_value < 0.05) and (metrics['max_delay_ms'] > 2)
        
        if metrics['is_traveling']:
            n_traveling += 1
        
        results.append(metrics)
    
    # Compute summary stats
    if results:
        mean_r2 = np.mean([r['r2'] for r in results])
        mean_delay = np.mean([r['max_delay_ms'] for r in results])
        pct_traveling = 100 * n_traveling / len(results)
    else:
        mean_r2, mean_delay, pct_traveling = 0, 0, 0
    
    # Event size distribution (diagnostic)
    event_sizes = [e['n_channels'] for e in events]
    print(f"  Event size distribution: {np.min(event_sizes)}-{np.max(event_sizes)} channels (median: {np.median(event_sizes):.0f})")
    print(f"  Traveling waves: {n_traveling}/{len(results)} ({pct_traveling:.1f}%)")
    
    # Save detailed results
    output_dir = trial_dir / "propagation_analysis_v2"
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
    
    return {
        'session': session,
        'trial': trial,
        'n_ripples': n_ripples,
        'n_channels': n_channels,
        'n_events': len(events),
        'n_traveling': n_traveling,
        'pct_traveling': pct_traveling,
        'mean_r2': mean_r2,
        'mean_delay': mean_delay
    }


def run_diagnostics(trial_dir: Path, z_low: float = 2.5):
    """Run diagnostic checks on the data."""
    print("\n" + "=" * 60)
    print("DIAGNOSTICS")
    print("=" * 60)
    
    events_df = load_ripple_events_with_peaks(trial_dir, z_low)
    if events_df is None:
        print("No events found!")
        return
    
    # 1. Time unit check
    print("\n1. TIME UNIT CHECK:")
    print(f"   Min peak time: {events_df['peak_ms'].min():.2f} ms")
    print(f"   Max peak time: {events_df['peak_ms'].max():.2f} ms")
    print(f"   Expected for 5-min trial: 0 - 300,000 ms")
    
    max_time = events_df['peak_ms'].max()
    if max_time > 1e6:
        print("   ⚠ WARNING: Times seem too large! Check units.")
    elif max_time < 1000:
        print("   ⚠ WARNING: Times seem too small! Check units.")
    else:
        print("   ✓ Times look reasonable")
    
    # 2. Inter-ripple interval distribution
    print("\n2. INTER-RIPPLE INTERVALS:")
    df_sorted = events_df.sort_values('peak_ms')
    iris = df_sorted['peak_ms'].diff().dropna()
    print(f"   Median IRI: {iris.median():.1f} ms")
    print(f"   Mean IRI: {iris.mean():.1f} ms")
    print(f"   5th percentile: {iris.quantile(0.05):.1f} ms")
    print(f"   Fraction < 30ms: {100 * (iris < 30).mean():.1f}%")
    
    # 3. Ripples per channel
    print("\n3. RIPPLES PER CHANNEL:")
    counts = events_df['channel'].value_counts()
    print(f"   Min: {counts.min()}, Max: {counts.max()}, Median: {counts.median():.0f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Propagation Analysis v2')
    parser.add_argument('--session', type=int, help='Session number')
    parser.add_argument('--trial', type=int, default=1, help='Trial number')
    parser.add_argument('--z_low', type=float, default=2.5)
    parser.add_argument('--window_ms', type=float, default=10.0, 
                        help='Event window size in ms (default: 10)')
    parser.add_argument('--min_channels', type=int, default=4,
                        help='Minimum channels for multi-channel event (default: 4)')
    parser.add_argument('--diagnostics', action='store_true',
                        help='Run diagnostics only')
    parser.add_argument('--batch', action='store_true',
                        help='Run on all completed sessions')
    args = parser.parse_args()
    
    if args.diagnostics and args.session:
        base = Path("/vol/brains/bd3/pesaranlab/Archie_RecStim_vSUBNETS220_2nd/matlab/mfiles/Chen")
        trial_dir = base / f"session{args.session:03d}" / f"trial{args.trial:03d}_bipolar"
        run_diagnostics(trial_dir, args.z_low)
        return
    
    if args.session and not args.batch:
        # Single session
        print("=" * 80)
        print(f"PROPAGATION ANALYSIS v2: Session {args.session}, Trial {args.trial}")
        print("=" * 80)
        
        result = analyze_session_propagation_v2(
            args.session, args.trial, args.z_low,
            window_ms=args.window_ms, min_channels=args.min_channels
        )
        
        if result:
            print(f"\nSUMMARY:")
            print(f"  Events: {result['n_events']}")
            print(f"  Traveling: {result['n_traveling']} ({result['pct_traveling']:.1f}%)")
            print(f"  Mean R²: {result['mean_r2']:.3f}")
            print(f"  Mean delay: {result['mean_delay']:.2f} ms")
        return
    
    if args.batch:
        # Batch mode - run on all sessions
        print("=" * 80)
        print("PROPAGATION ANALYSIS v2: Batch Mode")
        print("=" * 80)
        
        base = Path("/vol/brains/bd3/pesaranlab/Archie_RecStim_vSUBNETS220_2nd/matlab/mfiles/Chen")
        
        # Find all sessions with bipolar data
        sessions = []
        for d in sorted(base.glob("session[0-9][0-9][0-9]")):
            sess_num = int(d.name.replace("session", ""))
            trial_dir = d / f"trial{args.trial:03d}_bipolar"
            if trial_dir.exists():
                # Check if ripple data exists
                if any(trial_dir.glob("b[0-9][0-9][0-9]/ripples_*.mat")):
                    sessions.append(sess_num)
        
        print(f"Found {len(sessions)} sessions with ripple data")
        
        results = []
        for sess in sessions:
            print(f"\n[Session {sess}]")
            print("-" * 40)
            result = analyze_session_propagation_v2(
                sess, args.trial, args.z_low,
                window_ms=args.window_ms, min_channels=args.min_channels
            )
            if result:
                results.append(result)
        
        # Summary table
        if results:
            df = pd.DataFrame(results)
            
            print("\n" + "=" * 100)
            print("BATCH SUMMARY")
            print("=" * 100)
            print(f"\n{'Session':<10} {'Events':<10} {'Traveling':<12} {'%Travel':<10} {'Mean R²':<10} {'Mean Delay':<12}")
            print("-" * 74)
            
            for _, row in df.iterrows():
                print(f"{row['session']:<10} {row['n_events']:<10} {row['n_traveling']:<12} "
                      f"{row['pct_traveling']:.1f}%{'':<6} {row['mean_r2']:.3f}{'':<5} "
                      f"{row['mean_delay']:.2f} ms")
            
            print("-" * 74)
            print(f"{'TOTAL':<10} {df['n_events'].sum():<10} {df['n_traveling'].sum():<12} "
                  f"{100*df['n_traveling'].sum()/df['n_events'].sum():.1f}%")
            
            # Save
            df.to_csv(base / "propagation_summary_v2.csv", index=False)
            print(f"\nSaved: {base}/propagation_summary_v2.csv")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
