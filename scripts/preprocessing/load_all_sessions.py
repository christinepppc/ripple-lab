#!/usr/bin/env python3
"""
Load all valid sessions and trials based on duration and channel availability criteria.

Criteria:
  - Duration: 4.0 - 6.5 minutes
  - Channel availability: ≤ 10 missing channels (out of 220)
  - Sessions: 001 - 160
  - Format: session###/trial###/chan###/

Usage:
    python scripts/preprocessing/load_all_sessions.py --output_base_dir /path/to/output [--dry_run]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'packages' / 'ripple_core'))

import argparse
import numpy as np
import scipy.io as sio
from ripple_core.load import load_electrodes
import pandas as pd
from datetime import datetime


def check_trial_duration(session_idx: int, trial: int) -> tuple[bool, float, str]:
    """
    Check if trial duration is within acceptable range (4.0 - 6.5 minutes).
    
    Returns:
        (is_valid, duration_minutes, reason)
    """
    try:
        lfp = load_electrodes(session_idx, trial, 1)
        duration_sec = len(lfp) / 1000
        duration_min = duration_sec / 60
        
        if duration_min < 4.0:
            return False, duration_min, "too_short"
        elif duration_min > 6.5:
            return False, duration_min, "too_long"
        else:
            return True, duration_min, "ok"
    except Exception as e:
        return False, 0.0, f"error: {str(e)[:50]}"


def check_channel_availability(session_idx: int, trial: int) -> tuple[bool, int, int, str]:
    """
    Check how many channels are available for this trial.
    
    Returns:
        (is_valid, channels_available, channels_missing, reason)
    """
    available = 0
    missing = 0
    
    for ch in range(1, 221):
        try:
            lfp = load_electrodes(session_idx, trial, ch)
            if len(lfp) > 0:
                available += 1
            else:
                missing += 1
        except:
            missing += 1
    
    is_valid = missing <= 10
    reason = "ok" if is_valid else f"too_many_missing"
    
    return is_valid, available, missing, reason


def load_trial_data(
    session_idx: int, 
    trial: int, 
    output_base_dir: Path,
    available_channels: list[int]
) -> bool:
    """
    Load all available channels for a trial and save to proper directory structure.
    
    Returns:
        True if successful, False otherwise
    """
    session_dir = output_base_dir / f'session{session_idx:03d}'
    session_dir.mkdir(parents=True, exist_ok=True)
    
    trial_dir = session_dir / f'trial{trial:03d}'
    trial_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        for ch in available_channels:
            # Load data
            lfp = load_electrodes(session_idx, trial, ch)
            
            # Create channel directory
            chan_dir = trial_dir / f'chan{ch:03d}'
            chan_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to .mat file
            mat_file = chan_dir / f'sess{session_idx:03d}_trial{trial:03d}_chan{ch:03d}.mat'
            sio.savemat(str(mat_file), {
                'lfp': lfp,
                'fs': 1000,
                'channel': ch,
                'session': session_idx,
                'trial': trial
            })
        
        return True
    except Exception as e:
        print(f"  ✗ Error loading session {session_idx} trial {trial:03d}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Load all valid sessions/trials based on duration and channel criteria"
    )
    parser.add_argument(
        '--output_base_dir',
        type=str,
        default='/vol/brains/bd3/pesaranlab/Archie_RecStim_vSUBNETS220_2nd/matlab/mfiles/Chen',
        help='Base directory for output (default: standard Chen directory)'
    )
    parser.add_argument(
        '--session_range',
        type=int,
        nargs=2,
        default=[1, 160],
        help='Session range to scan [start, end] (default: 1 160)'
    )
    parser.add_argument(
        '--trial_range',
        type=int,
        nargs=2,
        default=[1, 30],
        help='Trial range to check per session [start, end] (default: 1 30)'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Only scan and report, do not load data'
    )
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        default=True,
        help='Skip trials that are already loaded (default: True)'
    )
    
    args = parser.parse_args()
    
    output_base_dir = Path(args.output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary tracking
    summary_data = []
    stats = {
        'total_checked': 0,
        'valid': 0,
        'loaded': 0,
        'skipped_duration': 0,
        'skipped_channels': 0,
        'skipped_exists': 0,
        'failed': 0,
    }
    
    print('=' * 100)
    print('SCANNING ALL SESSIONS FOR VALID TRIALS')
    print('=' * 100)
    print(f'\nCriteria:')
    print(f'  Duration: 4.0 - 6.5 minutes')
    print(f'  Max missing channels: 10 / 220')
    print(f'  Session range: {args.session_range[0]} - {args.session_range[1]}')
    print(f'  Trial range per session: {args.trial_range[0]} - {args.trial_range[1]}')
    print(f'  Output directory: {output_base_dir}')
    print(f'  Mode: {"DRY RUN" if args.dry_run else "LOADING DATA"}')
    print('=' * 100)
    
    start_time = datetime.now()
    
    # Scan all sessions
    for session_idx in range(args.session_range[0], args.session_range[1] + 1):
        print(f'\n{"="*100}')
        print(f'SESSION {session_idx:03d}')
        print(f'{"="*100}')
        
        session_valid_trials = []
        
        # Check all trials in this session
        for trial in range(args.trial_range[0], args.trial_range[1] + 1):
            stats['total_checked'] += 1
            
            # Check if already exists
            trial_dir = output_base_dir / f'session{session_idx:03d}' / f'trial{trial:03d}'
            if args.skip_existing and trial_dir.exists():
                # Check if it has channel directories
                chan_dirs = list(trial_dir.glob('chan???'))
                if len(chan_dirs) >= 200:  # Assume loaded if has many channel dirs
                    print(f'  Trial {trial:03d}: ⊙ Already loaded ({len(chan_dirs)} channels)')
                    stats['skipped_exists'] += 1
                    continue
            
            # Step 1: Check duration
            duration_ok, duration_min, duration_reason = check_trial_duration(session_idx, trial)
            
            if not duration_ok:
                if duration_reason not in ["error"]:
                    print(f'  Trial {trial:03d}: ✗ {duration_reason} ({duration_min:.2f} min)')
                stats['skipped_duration'] += 1
                
                summary_data.append({
                    'session': session_idx,
                    'trial': trial,
                    'status': 'skipped_duration',
                    'duration_min': duration_min,
                    'reason': duration_reason,
                    'channels_available': 0,
                    'channels_missing': 0,
                })
                continue
            
            # Step 2: Check channel availability
            print(f'  Trial {trial:03d}: Duration OK ({duration_min:.2f} min), checking channels...')
            channels_ok, ch_avail, ch_miss, ch_reason = check_channel_availability(session_idx, trial)
            
            if not channels_ok:
                print(f'    ✗ Too many missing channels ({ch_miss}/220 missing)')
                stats['skipped_channels'] += 1
                
                summary_data.append({
                    'session': session_idx,
                    'trial': trial,
                    'status': 'skipped_channels',
                    'duration_min': duration_min,
                    'reason': ch_reason,
                    'channels_available': ch_avail,
                    'channels_missing': ch_miss,
                })
                continue
            
            # Valid trial!
            stats['valid'] += 1
            print(f'    ✓ VALID ({ch_avail}/220 channels, {ch_miss} missing)')
            
            # Get list of available channels
            available_channels = []
            for ch in range(1, 221):
                try:
                    lfp = load_electrodes(session_idx, trial, ch)
                    if len(lfp) > 0:
                        available_channels.append(ch)
                except:
                    pass
            
            session_valid_trials.append({
                'trial': trial,
                'duration_min': duration_min,
                'channels_available': ch_avail,
                'channels_missing': ch_miss,
                'available_channels': available_channels
            })
            
            # Load data if not dry run
            if not args.dry_run:
                print(f'    Loading {ch_avail} channels...')
                success = load_trial_data(session_idx, trial, output_base_dir, available_channels)
                
                if success:
                    print(f'    ✓ Loaded to {trial_dir}/')
                    stats['loaded'] += 1
                    status = 'loaded'
                else:
                    print(f'    ✗ Failed to load')
                    stats['failed'] += 1
                    status = 'failed'
            else:
                status = 'valid_not_loaded'
            
            summary_data.append({
                'session': session_idx,
                'trial': trial,
                'status': status,
                'duration_min': duration_min,
                'reason': 'ok',
                'channels_available': ch_avail,
                'channels_missing': ch_miss,
            })
        
        # Summary for this session
        if len(session_valid_trials) > 0:
            print(f'\n  Session {session_idx:03d} summary: {len(session_valid_trials)} valid trials')
        else:
            print(f'\n  Session {session_idx:03d}: No valid trials found')
    
    # Overall summary
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print('\n' + '=' * 100)
    print('OVERALL SUMMARY')
    print('=' * 100)
    print(f'\nTime elapsed: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)')
    print(f'\nTrials checked: {stats["total_checked"]}')
    print(f'  Valid trials: {stats["valid"]}')
    print(f'  Loaded: {stats["loaded"]}')
    print(f'  Skipped (already exists): {stats["skipped_exists"]}')
    print(f'  Skipped (duration): {stats["skipped_duration"]}')
    print(f'  Skipped (missing channels): {stats["skipped_channels"]}')
    print(f'  Failed to load: {stats["failed"]}')
    
    # Save summary CSV
    summary_df = pd.DataFrame(summary_data)
    summary_file = output_base_dir / f'loading_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f'\nSummary saved to: {summary_file}')
    
    # Show breakdown by session
    if len(summary_df) > 0:
        valid_by_session = summary_df[summary_df['status'].isin(['valid_not_loaded', 'loaded'])].groupby('session').size()
        if len(valid_by_session) > 0:
            print(f'\n' + '=' * 100)
            print('VALID TRIALS BY SESSION')
            print('=' * 100)
            for session, count in valid_by_session.items():
                print(f'  Session {session:03d}: {count} valid trials')
    
    print('\n' + '=' * 100)
    print('DONE!')
    print('=' * 100)


if __name__ == "__main__":
    main()
